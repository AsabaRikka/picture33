' 无黑框启动智能切分工具
' 创建日期: 2026年4月27日

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' 获取当前脚本所在目录
scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Python 脚本路径
pythonScript = scriptPath & "\smart_split_gui.py"

' 使用 Pythonw 无窗口运行
WshShell.Run "pythonw """ & pythonScript & """", 0, False

' 清理对象
Set fso = Nothing
Set WshShell = Nothing