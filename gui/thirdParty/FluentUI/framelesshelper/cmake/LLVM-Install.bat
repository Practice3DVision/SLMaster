@echo off
setlocal
set IsTextMode=No

echo LLVM Installer [Version 0.1.0]
echo Based on VC-LTL installer. Copyright (C) Chuyu Team. All rights reserved.
echo.

if /i "%1"=="" goto TextMode
if /i "%1"=="/?" goto Help
if /i "%1"=="/I" goto Install
if /i "%1"=="/U" goto UnInstall

echo Invaild Command Parameter.
echo.
goto Help

:Install
set Root=%~dp0
set Root=%Root:~0,-1%
echo.
echo.Start writing to the registry
echo.[HKCU\Code\LLVM]
echo.Root="%Root%"
echo.
reg ADD HKCU\Code\LLVM /v Root /t REG_SZ /d "%Root%" /f
echo.
goto ExitScript

:UnInstall
echo.
reg delete HKCU\Code\LLVM /f
echo.
goto ExitScript

:Help
echo.Manages LLVM information in the current user registry.
echo.
echo.Install [Options]
echo.
echo.  /I    Add LLVM information to the current user registry.
echo.  /U    Remove LLVM information to the current user registry.
echo.  /?    Show this content.
goto ExitScript

:TextMode
set IsTextMode=Yes
echo.Welcome to use LLVM Installer! If you want to invoke this script
echo.silently, please use /? parameter to read help.
reg query HKCU\Code\LLVM /v Root >nul 2>nul && goto TextUnInstall || goto Install

:TextUnInstall
echo.
set /p Answer=LLVM appears to be installed, do you want to uninstall it (Y/N)?
if /i "%Answer%"=="Y" goto UnInstall
echo.
echo.Operation canceled.
echo.
goto ExitScript

:ExitScript
if /i "%IsTextMode%"=="Yes" pause
@echo on
