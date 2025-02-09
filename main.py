import re
import time
import os

# --- Лексер ---

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"{self.type}({self.value!r}) at {self.line}:{self.column}"

class Lexer:
    def __init__(self, code):
        self.code = code
        self.tokens = []

    def tokenize(self):
        tokens_spec = [
            ('CLASS',     r'\bclass\b'),
            ('NUMBER',    r'\d+(\.\d+)?'),
            ('IF',        r'\bif\b'),
            ('ELSE',      r'\belse\b'),
            ('PRINT',     r'\bprint\b'),
            ('TRUE',      r'\btrue\b'),
            ('FALSE',     r'\bfalse\b'),
            ('WHILE',     r'\bwhile\b'),
            ('FOR',       r'\bfor\b'),
            ('BREAK',     r'\bbreak\b'),
            ('CONTINUE',  r'\bcontinue\b'),
            ('FUNC',      r'\bfunc\b'),
            ('RETURN',    r'\breturn\b'),
            ('PUBLIC',    r'\bpublic\b'),
            ('PRIVATE',   r'\bprivate\b'),
            ('STATIC',    r'\bstatic\b'),
            ('PROTECTED', r'\bprotected\b'),
            ('USE',       r'\buse\b'),
            ('EQ',        r'=='),
            ('NE',        r'!='),
            ('LE',        r'<='), 
            ('GE',        r'>='),
            ('LT',        r'<'),
            ('GT',        r'>'),
            ('ASSIGN',    r'='),
            ('DOT',       r'\.'),
            ('LBRACE',    r'\{'),
            ('RBRACE',    r'\}'),
            ('SEMI',      r';'),
            ('STRING',    r'"[^"]*"'),
            ('MCOMMENT',  r'/\*[\s\S]*?\*/'),
            ('COMMENT',   r'//[^\n]*'),
            ('ID',        r'[A-Za-z_][A-Za-z0-9_]*'),
            ('OP',        r'[\+\-\*/]'),
            ('LPAREN',    r'\('),
            ('RPAREN',    r'\)'),
            ('COMMA',     r','),
            ('LSQUARE',   r'\['),
            ('RSQUARE',   r'\]'),
            ('NEWLINE',   r'\n'),
            ('SKIP',      r'[ \t]+'),
            ('MISMATCH',  r'.'),
        ]
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in tokens_spec)
        line_num = 1
        line_start = 0

        for mo in re.finditer(tok_regex, self.code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'SKIP' or kind == 'NEWLINE':
                if kind == 'NEWLINE':
                    line_num += 1
                    line_start = mo.end()
                elif '\n' in value:
                    line_num += value.count('\n')
                    line_start = mo.end() - value.rfind('\n')
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f"Неизвестный символ {value!r} на строке {line_num}")
            else:
                col = mo.start() - line_start + 1
                self.tokens.append(Token(kind, value, line_num, col))
        return self.tokens

# --- AST Узлы ---

class Unit:
    def __repr__(self):
        return "Unit()"

class MemberAccess:
    def __init__(self, object, member):
        self.object = object
        self.member = member

    def __repr__(self):
        return f"MemberAccess({self.object}, {self.member})"

class Num:
    def __init__(self, token):
        self.token = token
        self.value = float(token.value) if '.' in token.value else int(token.value)

    def __repr__(self):
        return f"Num({self.value})"

class Return:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Return({self.expr})"

class Bool:
    def __init__(self, token):
        self.token = token
        self.value = True if token.value == 'true' else False

    def __repr__(self):
        return f"Bool({self.value})"

class Var:
    def __init__(self, token):
        self.token = token
        self.value = token.value

    def __repr__(self):
        return f"Var({self.value})"

class BinOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinOp({self.left}, {self.op.value}, {self.right})"

class Assign:
    def __init__(self, left, right, modifiers=None):
        self.left = left
        self.right = right
        self.modifiers = modifiers or []

    def __repr__(self):
        return f"Assign(modifiers={self.modifiers}, {self.left}, {self.right})"

class Str:
    def __init__(self, token):
        self.token = token
        self.value = token.value[1:-1]

    def __repr__(self):
        return f"Str({self.value!r})"

class Print:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"Print({self.expr})"

class If:
    def __init__(self, condition, true_branch, false_branch=None):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return f"If({self.condition}), true_branch={self.true_branch}, false_branch={self.false_branch}"

class While:
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"While({self.condition}, body={self.body})"

class For:
    def __init__(self, init, condition, update, body):
        self.init = init
        self.condition = condition
        self.update = update
        self.body = body

    def __repr__(self):
        return f"For(init={self.init}, condition={self.condition}, update={self.update}, body={self.body})"

class FuncDef:
    def __init__(self, name, params, body, modifiers=None):
        self.name = name
        self.params = params
        self.body = body
        self.modifiers = modifiers or []

    def __repr__(self):
        return f"FuncDef(name={self.name}, modifiers={self.modifiers}, params={self.params}, body={self.body})"

class FuncCall:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __repr__(self):
        return f"FuncCall({self.func}, args={self.args})"

class Break:
    def __init__(self):
        pass
    def __repr__(self):
        return "Break()"

class Continue:
    def __init__(self):
        pass
    def __repr__(self):
        return "Continue()"

class ArrayLiteral:
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"ArrayLiteral({self.elements})"

class ArrayAccess:
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccess({self.array}, {self.index})"

class Use:
    def __init__(self, path_token, func_token):
        self.path_token = path_token
        self.func_token = func_token

    def __repr__(self):
        return f"Use(path={self.path_token.value}, func={self.func_token.value})"

class ClassDef:
    def __init__(self, name, body, modifiers=None):
        self.name = name
        self.body = body
        self.modifiers = modifiers or []

    def __repr__(self):
        return f"ClassDef(name={self.name}, modifiers={self.modifiers}, body={self.body})"

# --- Парсер ---

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat(self, token_type):
        token = self.current_token()
        if token is not None and token.type == token_type:
            self.pos += 1
            return token
        raise Exception(f"Ожидался токен {token_type}, но получен {token}")

    def parse(self):
        statements = []
        while self.current_token() is not None:
            if self.current_token().type == 'SEMI':
                self.eat('SEMI')
                continue
            stmt = self.statement()
            statements.append(stmt)
            if self.current_token() is not None and self.current_token().type == 'SEMI':
                self.eat('SEMI')
        return statements

    def parse_modifiers(self):
        mod_list = []
        while (self.current_token() is not None and
               self.current_token().type in ('PUBLIC', 'PRIVATE', 'STATIC', 'PROTECTED')):
            mod_list.append(self.current_token().value)
            self.eat(self.current_token().type)
        return mod_list

    def statement(self):
        modifiers = self.parse_modifiers()
        token = self.current_token()
        if token is None:
            return None

        if token.type == 'USE':
            return self.use_statement()
        elif token.type == 'PRINT':
            return self.print_statement()
        elif token.type == 'IF':
            return self.if_statement()
        elif token.type == 'WHILE':
            return self.while_statement()
        elif token.type == 'FOR':
            return self.for_statement()
        elif token.type == 'BREAK':
            self.eat('BREAK')
            return Break()
        elif token.type == 'CONTINUE':
            self.eat('CONTINUE')
            return Continue()
        elif token.type == 'RETURN':
            return self.return_statement()
        elif token.type == 'FUNC':
            func_node = self.func_def()
            func_node.modifiers = modifiers
            return func_node
        elif token.type == 'CLASS':
            class_node = self.class_def()
            class_node.modifiers = modifiers
            return class_node
        else:
            node = self.assignment_expr()
            return node

    def assignment_expr(self):
        node = self.expr()
        if self.current_token() is not None and self.current_token().type == 'ASSIGN':
            self.eat('ASSIGN')
            right = self.assignment_expr()
            return Assign(node, right)
        return node

    def use_statement(self):
        self.eat('USE')
        path_token = None
        func_token = None
        if self.current_token().type == 'STRING':
            path_token = self.eat('STRING')
        else:
            raise Exception("Ожидалась строка с путем к файлу для директивы use")
        if self.current_token().type == 'ID':
            func_token = self.eat('ID')
        else:
            raise Exception("Ожидался идентификатор функции после пути")
        self.eat('LPAREN')
        self.eat('RPAREN')
        return Use(path_token, func_token)

    def return_statement(self):
        self.eat('RETURN')
        expr = self.assignment_expr() if (self.current_token() is not None and
                                          self.current_token().type != 'SEMI') else None
        return Return(expr)

    def func_def(self):
        self.eat('FUNC')
        name_token = self.eat('ID')
        func_name = name_token.value
        self.eat('LPAREN')
        params = []
        if self.current_token() is not None and self.current_token().type == 'ID':
            params.append(self.eat('ID').value)
            while self.current_token() is not None and self.current_token().type == 'COMMA':
                self.eat('COMMA')
                params.append(self.eat('ID').value)
        self.eat('RPAREN')
        body = self.block()
        return FuncDef(func_name, params, body)

    def class_def(self):
        self.eat('CLASS')
        name_token = self.eat('ID')
        class_name = name_token.value
        class_body = self.block()
        return ClassDef(class_name, class_body)

    def block(self):
        if self.current_token() is not None and self.current_token().type == 'LBRACE':
            self.eat('LBRACE')
            statements = []
            while self.current_token() is not None and self.current_token().type != 'RBRACE':
                statements.append(self.statement())
                if self.current_token() is not None and self.current_token().type == 'SEMI':
                    self.eat('SEMI')
            self.eat('RBRACE')
            return statements
        else:
            return [self.statement()]

    def if_statement(self):
        self.eat('IF')
        if self.current_token() is not None and self.current_token().type == 'LPAREN':
            self.eat('LPAREN')
            condition = self.comprasion()
            self.eat('RPAREN')
        else:
            condition = self.comprasion()
        true_branch = self.block()
        false_branch = None
        if self.current_token() is not None and self.current_token().type == 'ELSE':
            self.eat('ELSE')
            false_branch = self.block()
        return If(condition, true_branch, false_branch)

    def while_statement(self):
        self.eat('WHILE')
        if self.current_token() is not None and self.current_token().type == 'LPAREN':
            self.eat('LPAREN')
            condition = self.comprasion()
            self.eat('RPAREN')
        else:
            condition = self.comprasion()
        body = self.block()
        return While(condition, body)

    def for_statement(self):
        self.eat('FOR')
        self.eat('LPAREN')
        init = None
        if self.current_token() is not None and self.current_token().type != 'SEMI':
            init = self.statement()
        self.eat('SEMI')
        condition = None
        if self.current_token() is not None and self.current_token().type != 'SEMI':
            condition = self.comprasion()
        self.eat('SEMI')
        update = None
        if self.current_token() is not None and self.current_token().type != 'RPAREN':
            update = self.statement()
        self.eat('RPAREN')
        body = self.block()
        return For(init, condition, update, body)

    def comprasion(self):
        node = self.expr()
        while (self.current_token() is not None and
               self.current_token().type in ('EQ', 'NE', 'LT', 'GT', 'LE', 'GE')):
            op_type = self.current_token().type
            op_token = self.eat(op_type)
            right = self.expr()
            node = BinOp(node, op_token, right)
        return node

    def print_statement(self):
        self.eat('PRINT')
        expr_node = self.assignment_expr()
        return Print(expr_node)

    def expr(self):
        node = self.term()
        while (self.current_token() is not None and
               self.current_token().type == 'OP' and
               self.current_token().value in ('+', '-')):
            op_token = self.eat('OP')
            right = self.term()
            node = BinOp(node, op_token, right)
        return node

    def term(self):
        node = self.factor()
        while (self.current_token() is not None and
               self.current_token().type == 'OP' and
               self.current_token().value in ('*', '/')):
            op_token = self.eat('OP')
            right = self.factor()
            node = BinOp(node, op_token, right)
        return node

    def primary(self):
        token = self.current_token()
        if token.type == 'NUMBER':
            self.eat('NUMBER')
            return Num(token)
        elif token.type == 'STRING':
            self.eat('STRING')
            return Str(token)
        elif token.type in ('TRUE', 'FALSE'):
            self.eat(token.type)
            return Bool(token)
        elif token.type == 'ID':
            id_token = self.eat('ID')
            if self.current_token() is not None and self.current_token().type == 'LPAREN':
                self.eat('LPAREN')
                args = []
                if self.current_token() is not None and self.current_token().type != 'RPAREN':
                    args.append(self.assignment_expr())
                    while self.current_token() is not None and self.current_token().type == 'COMMA':
                        self.eat('COMMA')
                        args.append(self.assignment_expr())
                self.eat('RPAREN')
                return FuncCall(Var(id_token), args)
            return Var(id_token)
        elif token.type == 'LPAREN':
            self.eat('LPAREN')
            if self.current_token() is not None and self.current_token().type == 'RPAREN':
                self.eat('RPAREN')
                return Unit()
            else:
                node = self.comprasion()
                self.eat('RPAREN')
                return node
        elif token.type == 'LSQUARE':
            return self.array_literal()
        else:
            raise Exception(f"Неожиданный токен: {token}")

    def array_literal(self):
        self.eat('LSQUARE')
        elements = []
        if self.current_token() is not None and self.current_token().type != 'RSQUARE':
            elements.append(self.assignment_expr())
            while self.current_token() is not None and self.current_token().type == 'COMMA':
                self.eat('COMMA')
                elements.append(self.assignment_expr())
        self.eat('RSQUARE')
        return ArrayLiteral(elements)

    def factor(self):
        node = self.primary()
        while True:
            token = self.current_token()
            if token is not None and token.type == 'LSQUARE':
                self.eat('LSQUARE')
                index_expr = self.assignment_expr()
                self.eat('RSQUARE')
                node = ArrayAccess(node, index_expr)
            elif token is not None and token.type == 'DOT':
                self.eat('DOT')
                member_token = self.eat('ID')
                node = MemberAccess(node, Var(member_token))
            else:
                break
        return node

# --- Исключения для управления потоком ---

class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# --- Поддержка классов ---

class ClassInstance:
    def __init__(self, blueprint, interpreter):
        self.blueprint = blueprint
        self.env = {}
        for stmt in blueprint.body:
            if isinstance(stmt, FuncDef):
                self.env[stmt.name] = stmt
        self.interpreter = interpreter

    def __repr__(self):
        return f"<Instance of class {self.blueprint.name}>"

# --- Интерпретатор ---

class Interpreter:
    def __init__(self, tree):
        self.tree = tree
        self.env = {}
        self.delay = 0

    def interpret(self):
        result = None
        for node in self.tree:
            result = self.visit(node)
            if self.delay > 0:
                time.sleep(self.delay / 1000.0)
        return result

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)

    def generic_visit(self, node):
        raise Exception(f"Нет метода visit_{type(node).__name__}")

    def visit_FuncDef(self, node):
        self.env[node.name] = node
        return node

    def visit_ClassDef(self, node):
        self.env[node.name] = node
        return node

    def visit_Num(self, node):
        return node.value

    def visit_Bool(self, node):
        return node.value

    def visit_Var(self, node):
        var_name = node.value
        if var_name in self.env:
            return self.env[var_name]
        else:
            raise Exception(f"Переменная {var_name} не определена")

    def visit_MemberAccess(self, node):
        obj = self.visit(node.object)
        if isinstance(obj, ClassInstance):
            member_name = node.member.value
            if member_name in obj.env:
                return obj.env[member_name]
            raise Exception(f"Member {member_name} not found in {obj}")
        raise Exception(f"Cannot access members of {type(obj)}")

    def visit_FuncCall(self, node):
        # Handle method calls on objects first
        if isinstance(node.func, MemberAccess):
            obj = self.visit(node.func.object)
            if isinstance(obj, ClassInstance):
                method_name = node.func.member.value
                if method_name in obj.env:
                    method = obj.env[method_name]
                    if isinstance(method, FuncDef):
                        args_values = [self.visit(arg) for arg in node.args]
                        if len(args_values) != len(method.params):
                            raise Exception("Incorrect number of arguments")
                        old_env = self.env
                        # Create a new environment for the method:
                        new_env = dict(obj.env)
                        # Bind method parameters and importantly bind "this" to the current instance
                        new_env["this"] = obj
                        for param, value in zip(method.params, args_values):
                            new_env[param] = value
                        self.env = new_env
                        result = None
                        try:
                            for stmt in method.body:
                                result = self.visit(stmt)
                        except ReturnException as ret:
                            result = ret.value
                        self.env = old_env
                        return result
                raise Exception(f"Method {method_name} not found in {obj}")
        # For normal function calls:
        func = self.visit(node.func)
        # If a class is "called", instantiate it.
        if isinstance(func, ClassDef):
            instance = ClassInstance(func, self)
            # Before returning the instance, call the constructor if it exists.
            if "init" in instance.env:
                init_method = instance.env["init"]
                args_values = [self.visit(arg) for arg in node.args]
                if len(args_values) != len(init_method.params):
                    raise Exception("Incorrect number of arguments in constructor")
                old_env = self.env
                new_env = dict(instance.env)
                new_env["this"] = instance  # Bind "this" for the constructor
                for param, value in zip(init_method.params, args_values):
                    new_env[param] = value
                self.env = new_env
                try:
                    for stmt in init_method.body:
                        self.visit(stmt)
                except ReturnException as ret:
                    pass  # Ignore returned value from constructor
                self.env = old_env
            return instance
        # Regular function call:
        if isinstance(func, FuncDef):
            args_values = [self.visit(arg) for arg in node.args]
            if len(args_values) != len(func.params):
                raise Exception("Incorrect number of arguments")
            old_env = self.env
            new_env = dict(self.env)
            for param, value in zip(func.params, args_values):
                new_env[param] = value
            self.env = new_env
            result = None
            try:
                for stmt in func.body:
                    result = self.visit(stmt)
            except ReturnException as ret:
                result = ret.value
            self.env = old_env
            return result
        raise Exception(f"{node.func} не является функцией для вызова")

    def visit_BinOp(self, node):
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)
        op = node.op.value
        if op == '+':
            if isinstance(left_val, str) or isinstance(right_val, str):
                return str(left_val) + str(right_val)
            return left_val + right_val
        elif op == '-':
            return left_val - right_val
        elif op == '*':
            return left_val * right_val
        elif op == '/':
            return left_val / right_val
        elif op == '==':
            return left_val == right_val
        elif op == '!=':
            return left_val != right_val
        elif op == '<':
            return left_val < right_val
        elif op == '>':
            return left_val > right_val
        elif op == '<=':
            return left_val <= right_val
        elif op == '>=':
            return left_val >= right_val
        else:
            raise Exception(f"Неизвестный оператор {op}")

    def visit_Assign(self, node):
        if isinstance(node.left, Var):
            var_name = node.left.value
            value = self.visit(node.right)
            self.env[var_name] = value
            return value
        elif isinstance(node.left, MemberAccess):
            obj = self.visit(node.left.object)
            if isinstance(obj, ClassInstance):
                member_name = node.left.member.value
                value = self.visit(node.right)
                obj.env[member_name] = value
                return value
            else:
                raise Exception(f"Неверное назначение для объекта {obj}")
        else:
            raise Exception("Левое значение не может участвовать в назначении")

    def visit_Print(self, node):
        value = self.visit(node.expr)
        print(value)
        return value

    def visit_Str(self, node):
        return node.value

    def visit_If(self, node):
        condition = self.visit(node.condition)
        if condition:
            for stmt in node.true_branch:
                self.visit(stmt)
        elif node.false_branch:
            for stmt in node.false_branch:
                self.visit(stmt)
        return None

    def visit_While(self, node):
        while self.visit(node.condition):
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except BreakException:
                break
            except ContinueException:
                continue
        return None

    def visit_For(self, node):
        if node.init is not None:
            self.visit(node.init)
        while True:
            if node.condition is not None and not self.visit(node.condition):
                break
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except BreakException:
                break
            except ContinueException:
                pass
            if node.update is not None:
                self.visit(node.update)
        return None

    def visit_Break(self, node):
        raise BreakException()

    def visit_Continue(self, node):
        raise ContinueException()

    def visit_Return(self, node):
        value = self.visit(node.expr) if node.expr is not None else None
        raise ReturnException(value)

    def visit_ArrayLiteral(self, node):
        return [self.visit(element) for element in node.elements]

    def visit_ArrayAccess(self, node):
        arr = self.visit(node.array)
        index = self.visit(node.index)
        if not isinstance(arr, list):
            raise Exception("Тип не является массивом для индексирования")
        if not isinstance(index, int):
            raise Exception("Индекс должен быть целым числом")
        try:
            return arr[index]
        except IndexError:
            raise Exception("Выход за пределы массива")

    def visit_Use(self, node):
        raw_path = node.path_token.value
        func_name = node.func_token.value
        if raw_path.startswith('"') and raw_path.endswith('"'):
            raw_path = raw_path[1:-1]
        if not os.path.exists(raw_path):
            raise Exception(f"Файл {raw_path} не найден")
        with open(raw_path, 'r', encoding='utf-8') as f:
            code2 = f.read()
        tokens2 = Lexer(code2).tokenize()
        parser2 = Parser(tokens2)
        tree2 = parser2.parse()
        interpreter2 = Interpreter(tree2)
        interpreter2.interpret()
        if func_name not in interpreter2.env:
            raise Exception(f"Функция {func_name} не найдена в {raw_path}")
        func_def = interpreter2.env[func_name]
        if not isinstance(func_def, FuncDef):
            raise Exception(f"{func_name} не является функцией")
        if len(func_def.params) != 0:
            raise Exception("Функция требует аргументы, а мы передаем 0")
        old_env = interpreter2.env
        new_env = dict(old_env)
        interpreter2.env = new_env
        result = None
        try:
            for stmt in func_def.body:
                result = interpreter2.visit(stmt)
        except ReturnException as ret:
            result = ret.value
        interpreter2.env = old_env
        return result

    def visit_Unit(self, node):
        return None

if __name__ == "__main__":
    # Пример демонстрации:
    code = """\
// enter your code here
"""
    lexer = Lexer(code)
    try:
        tokens = lexer.tokenize()
        print("Токены:")
        for tok in tokens:
            print(tok)
    except RuntimeError as e:
        print(e)
        tokens = []

    if tokens:
        parser = Parser(tokens)
        tree = parser.parse()
        print("\nAST:")
        for node in tree:
            print(node)
        interpreter = Interpreter(tree)
        interpreter.interpret()
        print("\nРезультаты выполнения (таблица символов):")
        for var, value in interpreter.env.items():
            print(f"{var} = {value}")