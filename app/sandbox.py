import subprocess

def run_command(command):
    """Run a shell command and return its output."""
    command = "docker exec -i monorepo-sandbox " + command
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"stdout": result.stdout.decode('utf-8'), "stderr": result.stderr.decode('utf-8'), "exit_code": result.returncode}
    except subprocess.CalledProcessError as e:
        return {"stdout": e.stdout.decode('utf-8'), "stderr": e.stderr.decode('utf-8'), "exit_code": e.returncode}