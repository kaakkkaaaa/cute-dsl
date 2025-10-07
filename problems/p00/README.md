## <b>Puzzle 0: Hello GPU</b>
### <b>Learning Objective</b>
Understand the basic structure of a CuTe DSL Program.
<br>
### <b>Challenge</b>
Write a kernel that prints "Hello from thread X" where X is the thread index. Launch it with 8 threads.
<br>

```py
def p00():
    """
    Write a kernel that prints "Hello from thread X" where X is the thread index.
    Launch it with 8 threads.
    """
    @cute.kernel
    def hello_kernel():
        # TODO: Get thread index
        # TODO: Print message with thread index
        pass

    @cute.jit
    def run_hello():
        # TODO: Launch kernel with 8 threads
        pass

    return run_hello
```

### <b>Key Concepts</b>:

`@cute.kernel`: Marks a function to run on GPU <br>
`@cute.jit`: Marks a function that can launch GPU code <br>
`cute.arch.thread_idx()`: Gets the current thread's index <br>
`.launch()`: Specifies grid/block dimensions <br>