from cx_Freeze import setup, Executable
import sys

# 依赖项
build_exe_options = {
    "packages": ["os", "random", "numpy", "matplotlib", "scipy", "cv2", "sklearn","typing","collections","PySimpleGUI","warnings","tkinter","traceback"],
    "include_files": []  # 如果有需要包含的数据文件或资源，可以在这里指定
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"  # 使用这个选项来隐藏控制台窗口

setup(
    name = "AppInner",
    version = "0.1",
    description = "My app_inner application!",
    options = {"build_exe": build_exe_options},
    executables = [Executable("app_inner.py", base=base)]
)
