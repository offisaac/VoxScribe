using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

[ComImport]
[Guid("00021401-0000-0000-C000-000000000046")]
internal class ShellLink
{
}

[ComImport]
[Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IPropertyStore
{
    [PreserveSig] int GetCount(out uint propertyCount);
    [PreserveSig] int GetAt(uint propertyIndex, out PropertyKey key);
    [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
    [PreserveSig] int SetValue(ref PropertyKey key, ref PropVariant value);
    [PreserveSig] int Commit();
}

[StructLayout(LayoutKind.Sequential, Pack = 4)]
internal struct PropertyKey
{
    public Guid FormatId;
    public uint PropertyId;

    public PropertyKey(Guid formatId, uint propertyId)
    {
        FormatId = formatId;
        PropertyId = propertyId;
    }
}

[StructLayout(LayoutKind.Explicit)]
internal struct PropVariant : IDisposable
{
    [FieldOffset(0)] private ushort valueType;
    [FieldOffset(8)] private IntPtr pointerValue;

    public PropVariant(string value)
    {
        valueType = 31;
        pointerValue = Marshal.StringToCoTaskMemUni(value);
    }

    public void Dispose()
    {
        if (pointerValue != IntPtr.Zero)
            Marshal.FreeCoTaskMem(pointerValue);
        pointerValue = IntPtr.Zero;
        valueType = 0;
    }
}

internal static class Program
{
    [STAThread]
    private static int Main(string[] arguments)
    {
        if (arguments.Length != 2)
            return 2;

        object link = new ShellLink();
        var persistFile = (IPersistFile)link;
        persistFile.Load(arguments[0], 2);
        var propertyStore = (IPropertyStore)link;
        var key = new PropertyKey(new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), 5);
        var value = new PropVariant(arguments[1]);
        try
        {
            int result = propertyStore.SetValue(ref key, ref value);
            if (result != 0)
                return 3;
            result = propertyStore.Commit();
            if (result != 0)
                return 4;
            persistFile.Save(arguments[0], true);
        }
        finally
        {
            value.Dispose();
            Marshal.FinalReleaseComObject(link);
        }
        return 0;
    }
}
