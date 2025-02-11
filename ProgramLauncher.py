import sys
import os
import time
import re

# Импортируем необходимые классы из модуля Hydra.
# Предполагается, что модуль Hydra содержит реализации Lexer, Parser и Interpreter.
from Hydra import Lexer, Parser, Interpreter

def run(file_path):
    """
    Запускает выполнение программы, написанной на языке Hydra, из файла с расширением .hy.
    
    Параметры:
        file_path (str): Путь к файлу с кодом на Hydra.
    """
    if not os.path.isfile(file_path):
        print(f"Файл не найден: {file_path}")
        return
    
    # Чтение исходного кода из файла.
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Прямое выполнение кода без токенизации и AST
    interpreter = Interpreter(code)
    interpreter.interpret()

def interactive_mode():
    """
    Режим интерактивного ввода команд.
    Поддерживает команды:
      run <путь_к_файлу> - запуск программы.
      exit              - выход из программы.
    """
    print("Запущен интерактивный режим. Введите 'run <путь_к_файлу>' для выполнения программы или 'exit' для выхода.")
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход из интерактивного режима.")
            break

        if not user_input:
            continue

        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()

        if command == "exit":
            print("Завершение работы.")
            break
        elif command == "run":
            if len(parts) < 2:
                print("Ошибка: необходимо указать путь к файлу. Пример: run example.hy")
            else:
                file_path = parts[1]
                run(file_path)
        else:
            print(f"Неизвестная команда: {command}. Используйте 'run <путь_к_файлу>' или 'exit'.")

if __name__ == "__main__":
    # Если скрипт запущен с аргументом командной строки "run", выполняем соответствующую логику.
    if len(sys.argv) >= 2:
        if sys.argv[1].lower() == "run":
            if len(sys.argv) >= 3:
                # Прямой запуск с указанием файла: python ProgramLauncher.py run <program.hy>
                run(sys.argv[2])
            else:
                # Если аргумент "run" указан, но файл не передан - запросим путь через input.
                file_path = input("Введите путь к файлу: ").strip()
                run(file_path)
        else:
            print(f"Неизвестная команда: {sys.argv[1]}. Поддерживаемая команда: run")
    else:
        # Если аргументы не указаны, запускаем интерактивный режим.
        interactive_mode()
