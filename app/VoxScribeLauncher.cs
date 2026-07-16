using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        string python = Path.Combine(root, "runtime", "Scripts", "pythonw.exe");
        string application = Path.Combine(root, "app", "voxscribe.py");

        if (!File.Exists(python) || !File.Exists(application))
        {
            MessageBox.Show(
                "VoxScribe 的运行环境不完整。\n\n安装目录：" + root,
                "VoxScribe",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
            return;
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = "\"" + application + "\"",
            WorkingDirectory = Path.Combine(root, "app"),
            UseShellExecute = false,
            CreateNoWindow = true
        };
        startInfo.EnvironmentVariables["HF_HOME"] = Path.Combine(root, "models", "huggingface");
        startInfo.EnvironmentVariables["HUGGINGFACE_HUB_CACHE"] = Path.Combine(root, "models", "huggingface", "hub");
        startInfo.EnvironmentVariables["PIP_CACHE_DIR"] = Path.Combine(root, "cache", "pip");
        startInfo.EnvironmentVariables["TEMP"] = Path.Combine(root, "cache", "temp");
        startInfo.EnvironmentVariables["TMP"] = Path.Combine(root, "cache", "temp");

        try
        {
            Process.Start(startInfo);
        }
        catch (Exception exception)
        {
            MessageBox.Show(
                "VoxScribe 启动失败：\n" + exception.Message,
                "VoxScribe",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }
}
