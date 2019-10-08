
def run_code(codePath,global_vars=None,local_vars=None):

  with open(codePath) as f: 
    code = compile(f.read(), codePath, 'exec') 
    exec(code,global_vars,local_vars)