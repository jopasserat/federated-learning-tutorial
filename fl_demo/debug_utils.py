import traceback
import os


def exception_message_with_line_number(e):
    stack_level = 1  # we start at the last line of the exception
    stack = None
    cwd = os.getcwd()
    # traverse the stack to grab the first application code that throw an exception
    while stack_level < 1000:
        stack = traceback.extract_tb(e.__traceback__, -stack_level)[0]
        if not stack.filename.startswith(cwd):
            stack_level += 1
        else:
            break
    filename, lineno, line = stack.filename, stack.lineno, stack.line
    return (
        type(e).__name__
        + "::"
        + str(e)
        + "@"
        + filename
        + ":"
        + str(lineno)
        + "  `"
        + line
        + "`"
    )
