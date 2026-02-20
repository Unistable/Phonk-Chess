@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Кодовая страница для корректного вывода русского текста в CMD
chcp 1251 >nul

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul 2>nul
if errorlevel 1 (
  echo Не удалось перейти в папку скрипта: %SCRIPT_DIR%
  pause
  exit /b 1
)

echo ==================================================
echo Phonk Chess — Быстрая настройка окружения (Windows)
echo ==================================================
echo.

rem 1) Проверка Python
where python >nul 2>nul
if errorlevel 1 (
  echo [ОШИБКА] Python не найден. Установите Python 3.8+ и повторите запуск.
  echo Ссылка: https://www.python.org/downloads/
  popd
  pause
  exit /b 1
)

rem 2) Создание виртуального окружения
if not exist ".venv" (
  echo [INFO] Создаю виртуальное окружение .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ОШИБКА] Не удалось создать .venv.
    popd
    pause
    exit /b 1
  )
) else (
  echo [INFO] Виртуальное окружение .venv уже существует.
)

set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
  echo [ОШИБКА] Не найден интерпретатор виртуального окружения: %VENV_PYTHON%
  popd
  pause
  exit /b 1
)

rem 3) Активация виртуального окружения (CMD)
echo [INFO] Активирую .venv ...
call ".venv\Scripts\activate.bat" 2>nul
if errorlevel 1 (
  echo [ПРЕДУПРЕЖДЕНИЕ] Не удалось активировать .venv через activate.bat.
  echo Для PowerShell используйте:
  echo   . .\.venv\Scripts\Activate.ps1
)

rem 4) Обновление pip и установка зависимостей
echo.
echo [INFO] Обновляю pip/setuptools/wheel ...
"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ПРЕДУПРЕЖДЕНИЕ] Не удалось обновить pip/setuptools/wheel.
)

if exist "requirements.txt" (
  echo [INFO] Устанавливаю зависимости из requirements.txt ...
  "%VENV_PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ПРЕДУПРЕЖДЕНИЕ] Установка зависимостей завершилась с ошибками.
  )
) else (
  echo [ПРЕДУПРЕЖДЕНИЕ] requirements.txt не найден в корне проекта.
)

rem 5) Поиск FFmpeg: PATH -> локальные папки ffmpeg* -> рекурсивный поиск
echo.
echo [INFO] Проверяю ffmpeg.exe ...
set "FFMPEG_BIN="
where ffmpeg >nul 2>nul
if errorlevel 1 (
  rem 5.1) Локальные папки внутри проекта
  for /d %%D in ("ffmpeg*" "*ffmpeg*") do (
    if not defined FFMPEG_BIN if exist "%%~fD\bin\ffmpeg.exe" set "FFMPEG_BIN=%%~fD\bin"
  )

  rem 5.2) Частые абсолютные пути: C:\ffmpeg*\bin, Downloads, Desktop, Program Files
  if not defined FFMPEG_BIN (
    for /f "delims=" %%F in ('dir /b /s /a:-d "%SystemDrive%\ffmpeg*\bin\ffmpeg.exe" 2^>nul') do (
      if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
  )

  if not defined FFMPEG_BIN (
    for /f "delims=" %%F in ('dir /b /s /a:-d "%USERPROFILE%\Downloads\ffmpeg*\bin\ffmpeg.exe" 2^>nul') do (
      if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
  )

  if not defined FFMPEG_BIN (
    for /f "delims=" %%F in ('dir /b /s /a:-d "%USERPROFILE%\Desktop\ffmpeg*\bin\ffmpeg.exe" 2^>nul') do (
      if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
  )

  if not defined FFMPEG_BIN (
    for /f "delims=" %%F in ('dir /b /s /a:-d "%ProgramFiles%\ffmpeg*\bin\ffmpeg.exe" 2^>nul') do (
      if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
  )

  if not defined FFMPEG_BIN (
    for /f "delims=" %%F in ('dir /b /s /a:-d "%ProgramFiles(x86)%\ffmpeg*\bin\ffmpeg.exe" 2^>nul') do (
      if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
  )

  rem 5.3) Переменная окружения FFMPEG_HOME (если задана)
  if not defined FFMPEG_BIN if defined FFMPEG_HOME (
    if exist "%FFMPEG_HOME%\bin\ffmpeg.exe" set "FFMPEG_BIN=%FFMPEG_HOME%\bin\"
  )

  rem 5.4) Общий fallback внутри проекта
  if not defined FFMPEG_BIN (
    for /f "delims=" %%F in ('dir /b /s /a:-d "*ffmpeg*\bin\ffmpeg.exe" 2^>nul') do (
      if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpF"
    )
  )

  if defined FFMPEG_BIN (
    set "PATH=!FFMPEG_BIN!;%PATH%"
    echo [OK] Найден FFmpeg: !FFMPEG_BIN!ffmpeg.exe
    echo [OK] Путь добавлен в PATH для текущей сессии.
  ) else (
    echo [ПРЕДУПРЕЖДЕНИЕ] FFmpeg не найден.
    echo Скачайте FFmpeg и добавьте путь к bin в системный PATH или положите в папку проекта.
    echo Полезные ссылки:
    echo   https://www.gyan.dev/ffmpeg/builds/
    echo   https://ffmpeg.org/download.html
  )
) else (
  echo [OK] ffmpeg уже доступен в PATH.
)

rem 6) Поиск Stockfish: корень -> папки stockfish* -> рекурсивный поиск
echo.
echo [INFO] Проверяю stockfish.exe ...
set "STOCKFISH_EXE="

if exist "stockfish.exe" set "STOCKFISH_EXE=%CD%\stockfish.exe"

if not defined STOCKFISH_EXE (
  for /d %%D in ("stockfish*" "*stockfish*") do (
    if not defined STOCKFISH_EXE if exist "%%~fD\stockfish.exe" set "STOCKFISH_EXE=%%~fD\stockfish.exe"
  )
)

if not defined STOCKFISH_EXE (
  for /f "delims=" %%F in ('dir /b /s /a:-d "*stockfish*\stockfish.exe" 2^>nul') do (
    if not defined STOCKFISH_EXE set "STOCKFISH_EXE=%%~fF"
  )
)

if defined STOCKFISH_EXE (
  echo [OK] Найден Stockfish: !STOCKFISH_EXE!
) else (
  echo [ПРЕДУПРЕЖДЕНИЕ] stockfish.exe не найден.
  echo Скачать можно здесь: https://stockfishchess.org/download/
  echo После распаковки укажите путь через параметр --stockfish.
)

rem 7) Итоговые инструкции
echo.
echo ==================================================
echo Готово
echo ==================================================
echo Пример запуска:
if defined STOCKFISH_EXE (
  echo   python main.py --pgn input_game.pgn --audio music.mp3 --output phonk_edit.mp4 --stockfish "!STOCKFISH_EXE!"
) else (
  echo   python main.py --pgn input_game.pgn --audio music.mp3 --output phonk_edit.mp4
)
echo.
echo Подсказки:
echo - Для PowerShell: . .\.venv\Scripts\Activate.ps1
echo - Для постоянного FFmpeg: добавьте путь к папке bin в системный PATH.

echo.
echo Завершено. Нажмите любую клавишу для выхода.
pause >nul

popd
endlocal