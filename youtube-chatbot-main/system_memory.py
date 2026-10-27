"""Total and available RAM in MiB on Windows, Linux and macOS, without extra packages."""
import ctypes
import os
import re
import subprocess
import sys


def _windows():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ['total', 'available', 'page', 'pagefree', 'virtual', 'virtualfree', 'extended']]
    sample = Memory()
    sample.length = ctypes.sizeof(sample)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(sample)):
        raise OSError('Memory inspection failed.')
    return sample.total // 2**20, sample.available // 2**20


def _linux():
    values = {}
    with open('/proc/meminfo', encoding='ascii') as handle:
        for line in handle:
            name, _, rest = line.partition(':')
            values[name] = int(rest.split()[0])  # kB
    # MemAvailable counts reclaimable cache too, unlike MemFree.
    available = values.get('MemAvailable', values.get('MemFree', 0))
    return values['MemTotal'] // 1024, available // 1024


def _macos():
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True,
                               text=True, check=True).stdout)
    stats = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r'page size of (\d+)', stats).group(1))
    pages = sum(int(re.search(rf'{name}:\s+(\d+)', stats).group(1))
                for name in ['Pages free', 'Pages inactive', 'Pages speculative'])
    return total // 2**20, pages * page // 2**20


def memory_mib():
    """Return (total, available) RAM in MiB."""
    if os.name == 'nt':
        return _windows()
    if sys.platform == 'darwin':
        return _macos()
    return _linux()
