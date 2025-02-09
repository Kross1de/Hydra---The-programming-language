# Hydra---The-programming-language
New programming language written on 'Python' named Hydra

## Introduction
Hydra is a lightweight, dynamically typed language defined by the lexical, syntactical, and interpretive rules in the provided code. It features classes, functions, arrays, branching (if/else), loops (while, for), and other common structures typical of modern languages. This documentation explains Hydra’s key features and how they are processed by the lexer, parser, and interpreter.

---

## Lexical Structure

### Tokens
Hydra is tokenized by a set of regular expressions, each representing a specific type of token. Some notable tokens include:

• CLASS, IF, ELSE, PRINT, TRUE, FALSE, WHILE, FOR, BREAK, CONTINUE, FUNC, RETURN, USE  
• PUBLIC, PRIVATE, STATIC, PROTECTED (class/function modifiers)  
• EQ (==), NE (!=), LT (<), GT (>), LE (<=), GE (>=)  
• ASSIGN (=), DOT (.), LBRACE ({), RBRACE (}), LPAREN ((), RPAREN ()), LSQUARE ([]), RSQUARE (])  
• STRING (quoted text, e.g., "hello"), ID (identifier for variables/functions/classes), NUMBER (integer or float), OP (+,-,*,/)  

Whitespace and newlines are either treated as ignorable (SKIP) or as line-break tokens (NEWLINE) where appropriate. Unrecognized characters cause an error.

### Comments
Hydra supports two kinds of comments:
• Single-line comments with //  
• Multiline comments with /* ... */

---

## Syntax and Parsing

### Basic Program Structure
A Hydra program is composed of statements, each ending with an optional semicolon (;). These statements can contain variable assignments, function definitions, class definitions, if statements, loops, and others.

### Variables and Assignments
You can create a variable simply by assigning a value to an identifier:

    x = 5;
    y = "Hello";

Hydra does not require a keyword to declare a variable; any identifier that appears on the left side of = is treated as a variable name.

### Data Types
The language supports:
• Numbers: integer (e.g., 42) or float (e.g., 3.14).  
• Strings: enclosed in double quotes (e.g., "Hello world").  
• Booleans: true or false.  
• Arrays: declared with square brackets, e.g., [1,2,3].  

### Operators
#### Arithmetic
• +, -, *, /  
• When using + on strings, it performs concatenation.  

#### Comparison
• ==, !=, <, >, <=, >=  

### Conditionals (If/Else)
An if statement can appear in two forms:

    if(condition) {
        // true-branch
    }

or

    if(condition) {
        // true-branch
    } else {
        // false-branch
    }

Conditions are evaluated as truthy or falsy according to standard Booleans.

### Loops
Hydra offers two main loop constructs: while and for.

#### While Loop
    while(condition) {
        // loop body
    }

The loop continues until the condition is falsy.

#### For Loop
    for(initializer; condition; updater) {
        // loop body
    }

Initialization typically assigns or declares variables. The condition is checked each iteration to decide whether to continue. Finally, the updater runs at the end of each iteration.

### Break and Continue
• break exits the current loop immediately.  
• continue jumps to the next iteration of the loop.

### Functions
Declaration syntax:

    func functionName(param1, param2) {
        // body
        return someValue;
    }

Functions may include zero, one, or multiple parameters. A return statement ends the function’s execution and optionally returns a value. Functions can be called with the same number of arguments as parameters.

Example usage:
    
    func greet(name) {
        print("Hello, " + name);
    }

    greet("Hydra");

### Classes
Hydra supports basic OOP constructs. Class definitions begin with the class keyword, followed by a class body enclosed in braces:

    class MyClass {
        // statements, including function definitions
    }

Members can include variable assignments, methods (using func), or other statements. Classes support the modifiers public, private, static, and protected.

#### Instantiation
Calling a class by name acts like a constructor call:

    let instance = MyClass();

If the class contains a func named init, that method is called automatically upon instantiation. Inside methods, this references the instance.

### Member Access
Use the . operator on instances:

    instance.memberName

### Arrays
Arrays are declared with square brackets:

    [element1, element2, ...]

Use array[index] to access or assign values. If index is out of range or not an int, an error is thrown.

### The Use Statement
Hydra can import and run code from another file using the use keyword:

    use "pathToFile" someFunctionName();

This directive loads and interprets another Hydra source file at runtime, then invokes the specified function within it.

---

## Interpreter Semantics
The interpreter:
1. Maintains a global environment mapping variable names to values.  
2. Binds function names to their definitions.  
3. Evaluates statements sequentially.  
4. Throws exceptions to manage break, continue, and return flows.  
5. Resolves array, class, and function calls with object-bound environments.

Execution flow is handled by the interpreter, which looks for special methods if using classes and calls them when appropriate (e.g., init for constructors).

---

## Example of a Simple Program
Below is a minimal Hydra example:

    func main() {
        x = 10;
        print("x before loop: " + x);
        
        for(i = 0; i < x; i = i + 1) {
            if(i == 5) {
                print("Halfway there!");
            }
        }
        
        print("Done!");
    }

Running this would:
• Define the function main.  
• In main, it initializes x to 10, prints its value, runs a for loop from i = 0 to i < 10, and prints a special message at i = 5. Finally, it prints "Done!".

---

## Conclusion
Hydra’s design offers a straightforward combination of functional and object-oriented programming features. It retains flexible syntax and runtime interpretation, making it suitable for quick scripting and prototyping. The code provided demonstrates the full implementation of the lexer, parser, and interpreter and serves as the reference for the language’s current specification.
