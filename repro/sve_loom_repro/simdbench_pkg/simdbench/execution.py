from typing import Optional, Callable, Dict, List, Sequence
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
import re
import subprocess
import json
import psutil
from simdbench.global_var import *

def clean_generated_code(output_str: str, entry_point: str) -> str:
    # Regular expression to match Markdown code blocks with optional language tag
    # code_block_pattern = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)
    code_block_pattern = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
    matches = code_block_pattern.finditer(output_str)
    res = output_str
    for match in matches:
        code = match.group(1)
        if entry_point in code:
            res =  code.strip()
            break 
    return res

def clean_error_message(error_msg):
    """
    clean the path in the error message, keeping the filename.
    example:
    input: "/home/XXX/main.cpp:15: error"
    output: "[REDACTED]/main.cpp:15: error"
    """
    sanitized = re.sub(
        r'(?P<quote>["\'])?/home[\w\s./-]+/(?P<filename>[^/\s]+)(?P=quote)?',
        lambda m: (m.group("quote") or "") + "[REDACTED]/" + m.group("filename") + (m.group("quote") or ""),
        error_msg,
        flags=re.ASCII
    )
    sanitized = re.sub(
        r'(?P<quote>["\'])?/mnt[\w\s./-]+/(?P<filename>[^/\s]+)(?P=quote)?',
        lambda m: (m.group("quote") or "") + "[REDACTED]/" + m.group("filename") + (m.group("quote") or ""),
        sanitized,
        flags=re.ASCII
    )
    return sanitized


def _safe_decode_process_output(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", errors="replace")


ROOT = os.path.dirname(os.path.abspath(__file__))
benchmark_src = os.path.join(ROOT, "../", "benchmark/build/src")
benchmark_include = os.path.join(ROOT, "../", "benchmark/include")
utils = os.path.join(ROOT, "../", "simdbench/utils")
GENERIC_CXX_HEADER = get_header()
GENERIC_C_HEADER = "\n".join([
    "#include <stddef.h>",
    "#include <stdint.h>",
    "#include <float.h>",
    "#include <string.h>",
    "#include <math.h>",
    "#include <ctype.h>",
    "#include <limits.h>",
    "#include <stdio.h>",
    "#include <stdlib.h>",
    "",
])

COMPILE_MODE_SPECS = {
    "cxx17": {
        "source_ext": ".cpp",
        "compile_args": ["-std=c++17"],
    },
    "c11": {
        "source_ext": ".c",
        "compile_args": ["-x", "c", "-std=gnu11"],
    },
}


def normalize_compile_modes(raw_modes) -> List[str]:
    modes: List[str] = []
    if raw_modes is None:
        return ["cxx17"]
    if isinstance(raw_modes, str):
        items: Sequence[str] = [part.strip() for part in raw_modes.split(",")]
    elif isinstance(raw_modes, (list, tuple)):
        items = [str(part).strip() for part in raw_modes]
    else:
        return ["cxx17"]

    for mode in items:
        if mode in COMPILE_MODE_SPECS and mode not in modes:
            modes.append(mode)
    return modes or ["cxx17"]


def adapt_code_for_compile_mode(cpp_code: str, compile_mode: str) -> str:
    if compile_mode != "c11":
        return cpp_code
    if GENERIC_CXX_HEADER and cpp_code.startswith(GENERIC_CXX_HEADER):
        return GENERIC_C_HEADER + cpp_code[len(GENERIC_CXX_HEADER):]
    if GENERIC_CXX_HEADER:
        return cpp_code.replace(GENERIC_CXX_HEADER, GENERIC_C_HEADER, 1)
    return cpp_code


def exec_cpp(cpp_code: str, intrinsic: str, timeout: float, googlebench: bool=False,
             optimization = "-O0", compile_modes = None) -> tuple:
    '''
    0: something failed
    1: compilation and execution success
    '''
    with tempfile.TemporaryDirectory(prefix='simdbench_', dir=ROOT) as tmp_dir:
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        
        dt = tempfile.mktemp(dir='')
        compile_modes = normalize_compile_modes(compile_modes)
        exe_path = os.path.join(tmp_dir, f"{dt}.out")
        last_error = ""
        compile_timeout = max(timeout * 0.25, 0.1)
        run_timeout = max(timeout * 0.7, 0.1)

        for compile_mode in compile_modes:
            mode_spec = COMPILE_MODE_SPECS[compile_mode]
            cpp_path = os.path.join(tmp_dir, f"{dt}_{compile_mode}{mode_spec['source_ext']}")
            compile_code = adapt_code_for_compile_mode(cpp_code, compile_mode)
            with open(cpp_path, "w", encoding="utf-8") as file:
                file.write(compile_code)

            compilation_cmd = [
                cpp_compilers[intrinsic],
                *mode_spec["compile_args"],
                cpp_path,
                *intrin_flags[intrinsic].split(),
                f"-I{utils}",
                optimization
            ]
            if(googlebench): compilation_cmd += f" -L{benchmark_src} -isystem {benchmark_include} -lbenchmark -lpthread ".split(' ')
            compilation_cmd += ["-o", exe_path]
            compilation_res = subprocess.run(
                compilation_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=compile_timeout,
            )
            compilation_stderr = _safe_decode_process_output(compilation_res.stderr)
            if compilation_res.returncode != 0:
                last_error = f"[compile_mode={compile_mode}] compilation failed: {clean_error_message(compilation_stderr)}"
                continue

            execution_cmd = [exe_path]
            emulator = emulators[intrinsic] # None: no emulator needed
            if(emulator!=None): execution_cmd = [emulator] + execution_cmd
            if(googlebench): execution_cmd += ["--benchmark_format=json"]
            print(f"the execution cmd = {execution_cmd}", flush=True)
            execution_res = subprocess.run(
                execution_cmd,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=run_timeout,
            )
            execution_stdout = _safe_decode_process_output(execution_res.stdout)
            execution_stderr = _safe_decode_process_output(execution_res.stderr)
            if execution_res.returncode != 0:
                last_error = f"[compile_mode={compile_mode}] runtime failed: {clean_error_message(execution_stderr)}"
                continue

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

            return (1, execution_stdout)

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir

        return (0, last_error if last_error else "compilation failed: no compile mode attempted")


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        if parent.is_running():
            children = parent.children(recursive=True)
            for child in children:
                if child.is_running():
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            if parent.is_running():
                parent.kill()
    except psutil.NoSuchProcess:
        pass

def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():
        # Construct the check program and run it.
        intrinsic = problem['intrinsic']
        compile_modes = problem.get("compile_modes")
        if(completion.find(problem['entrypoint_simd']) == -1 and intrinsic == "scalar"):
            # scalar execution for reusing simd test suites
            _completion = completion.replace(problem['entrypoint_scalar'], problem['entrypoint_simd'])
        else: _completion = completion
        candidate_prelude = str(problem.get("candidate_prelude") or problem.get("completion_prelude") or "")
        generated_code = clean_generated_code(_completion, problem['entrypoint_simd'])
        
        correctness_cpp_no_simd_header = (
            get_header() + '\n' + \
            problem['solution_scalar'] + '\n' + \
            candidate_prelude + '\n' + \
            generated_code + '\n' + \
            problem['test_correctness']
        )
        headers = get_header(intrinsic).split('\n')
        for header in headers:
            if(header.startswith("#include")):
                # remove the header from the correctness_cpp_no_simd_header
                correctness_cpp_no_simd_header = correctness_cpp_no_simd_header.replace(header, "")

        correctness_cpp = (
            get_header() + '\n' + \
            get_header(intrinsic) + '\n' + problem['solution_scalar'] + '\n' + \
            candidate_prelude + '\n' + \
            generated_code + '\n' + \
            problem['test_correctness']
        )
        # check whether the completion contains intrinsic
        # by removing the intrinsic header
        if intrinsic != "scalar":
            try:
                with swallow_io():
                    with time_limit(timeout):
                        # WARNING: This program exists to execute untrusted model-generated code.
                        correctness_res, correctness_stdout = exec_cpp(
                            correctness_cpp_no_simd_header,
                            'scalar',
                            timeout,
                            compile_modes=compile_modes,
                        )
                if(correctness_res == 1): # compilation success without intrinsic header
                    result.append("no intrinsic in code")
                    return
            except:
                pass
        
        try:
            with swallow_io():
                with time_limit(timeout):
                    # WARNING: This program exists to execute untrusted model-generated code.
                    correctness_res, correctness_stdout = exec_cpp(
                        correctness_cpp,
                        intrinsic,
                        timeout,
                        compile_modes=compile_modes,
                    )
            if(correctness_res == 0):
                result.append(f"{correctness_stdout}")
            else: # 1
                data = json.loads(correctness_stdout)
                if(data['correctness']==0):
                    result.append("logical bug")
                else:
                    result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        kill_process_tree(p.pid)

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 


def estimate_speedup(speedups, delta:float = 0.2) -> float:
    """
    for each problem
    """
    if type(speedups) is not dict:
        return -1.0
    tbd = []
    for key, value in speedups.items():
        tbd.append(value["speedup"])
    tbd = sorted(tbd)
    n = len(tbd)
    speedups_filtered = tbd[int(n*delta):n-int(n*delta)]
    if len(speedups_filtered) == 0:
        return sum(tbd) / n
    return sum(speedups_filtered) / len(speedups_filtered)


def check_perfomance(problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None, 
                     optimization = "-O0", n_reputation: int = 5) -> Dict:
    """
    Evaluates the performance of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():
        # Construct the check program and run it.
        intrinsic = problem['intrinsic']
        compile_modes = problem.get("compile_modes")
        if(completion.find(problem['entrypoint_simd']) == -1 and intrinsic == "scalar"):
            # scalar execution for reusing simd test suites
            _completion = completion.replace(problem['entrypoint_scalar'], problem['entrypoint_simd'])
        else: _completion = completion
        candidate_prelude = str(problem.get("candidate_prelude") or problem.get("completion_prelude") or "")
        generated_code = clean_generated_code(_completion, problem['entrypoint_simd'])
        
        # Functional correctness check
        data = {}
        data['correctness'] = 0
        correctness_cpp_no_simd_header = (
            get_header() + '\n' + \
            problem['solution_scalar'] + '\n' + \
            candidate_prelude + '\n' + \
            generated_code + '\n' + \
            problem['test_correctness']
        )
        headers = get_header(intrinsic).split('\n')
        for header in headers:
            if(header.startswith("#include")):
                # remove the header from the correctness_cpp_no_simd_header
                correctness_cpp_no_simd_header = correctness_cpp_no_simd_header.replace(header, "")

        correctness_cpp = (
            get_header() + '\n' + \
            get_header(intrinsic) + '\n' + problem['solution_scalar'] + '\n' + \
            candidate_prelude + '\n' + \
            generated_code + '\n' + \
            problem['test_correctness']
        )
        # check whether the completion contains intrinsic
        # by removing the intrinsic header
        try:
            with swallow_io():
                with time_limit(timeout):
                    # WARNING: This program exists to execute untrusted model-generated code.
                    correctness_res, correctness_stdout = exec_cpp(
                        correctness_cpp_no_simd_header,
                        'scalar',
                        timeout,
                        compile_modes=compile_modes,
                    )
            if(correctness_res == 1): # compilation success without intrinsic header
                result.append("no intrinsic in code")
                return
        except:
            pass
        try:
            with swallow_io():
                with time_limit(timeout):
                    # WARNING: This program exists to execute untrusted model-generated code.
                    correctness_res, correctness_stdout = exec_cpp(
                        correctness_cpp,
                        intrinsic,
                        timeout,
                        optimization=optimization,
                        compile_modes=compile_modes,
                    )
            if(correctness_res == 0):
                result.append(-1)
                return
            else: # 1
                data = json.loads(correctness_stdout)
                if(data['correctness']==0):
                    result.append(-1)
                    return
                else:
                    pass
        except TimeoutException:
            result.append(-1)
        except BaseException as e:
            result.append(-1)
        
        
        # Performance evaluation
        correctness_passed = (data['correctness']==1)
        performance_cpp = (
            "#include <benchmark/benchmark.h>\n" + get_header() + '\n' + \
            get_header(intrinsic) + '\n' + problem['solution_scalar'] + '\n' + \
            candidate_prelude + '\n' + \
            generated_code + '\n' + \
            problem['test_performance']
        )
        if correctness_passed != True:
            return
        else:
            try:
                speedups = {}
                for iteration in range(n_reputation):
                    with swallow_io():
                        with time_limit(timeout):
                            # WARNING: This program exists to execute untrusted model-generated code.
                            performance_res, performance_stdout = exec_cpp(performance_cpp, intrinsic, timeout, 
                                                                        googlebench=True, optimization=optimization)
                    if(performance_res == 0):
                        result.append(performance_stdout)
                    else: # 1
                        data2 = json.loads(performance_stdout)
                        tbd = {}
                        name_lst = set()
                        for item in data2['benchmarks']:
                            if(item['name'].startswith("Scalar/")):
                                name = item['name'].replace("Scalar/", "") + f"-{iteration}"
                                name_lst.add(name)
                            elif(item['name'].startswith("SIMD/")):
                                name = item['name'].replace("SIMD/", "") + f"-{iteration}"
                                name_lst.add(name)
                        for name in name_lst:
                            tbd[name] = {"scalar": 0, "simd": 0, "speedup": 0}
                        for item in data2['benchmarks']:
                            if(item['name'].startswith("Scalar/")):
                                name = item['name'].replace("Scalar/", "") + f"-{iteration}"
                                tbd[name]["scalar"] = item['real_time']
                            elif(item['name'].startswith("SIMD/")):
                                name = item['name'].replace("SIMD/", "") + f"-{iteration}"
                                tbd[name]["simd"] = item['real_time']
                        for key in tbd:
                            tbd[key]["speedup"] = tbd[key]["scalar"] / tbd[key]["simd"]
                        speedups = Merge(speedups, tbd)
                
                if len(speedups) == 0:
                    result.append(-2)
                else:
                    result.append(speedups)

            except Exception as e:
                result.append(e.__str__())

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        kill_process_tree(p.pid)

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        speedup = estimate_speedup(result[0]) if isinstance(result[0], dict) else -1.0,
        result=result[0] if result[0] != -1 else "correctness failed",
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
