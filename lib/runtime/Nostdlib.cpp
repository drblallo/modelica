#ifdef WINDOWS_NOSTDLIB
#include <Windows.h>
#include "marco/runtime/Nostdlib.h"
#include "marco/runtime/UtilityFunctions.h"
#include "marco/runtime/Runtime.h"

extern "C" int __main()
{
	//runSimulation();
	return 0;
}

BOOL WINAPI DllMain(
    HINSTANCE hinstDLL,
    DWORD fdwReason,
    LPVOID lpReserved)
{
    return TRUE;
}

extern "C" BOOL WINAPI DllMainCRTStartup(
	HINSTANCE hinstDLL,
	DWORD fdwReason,
	LPVOID lpvReserved)
{
	return TRUE;
}

extern "C" BOOL WINAPI _DllMainCRTStartup(
    HINSTANCE hinstDLL,
    DWORD fdwReason,
    LPVOID lpvReserved)
{
  return TRUE;
}

#ifdef MSVC_BUILD
extern "C" __declspec(noreturn) void __cdecl 
__imp__invalid_parameter_noinfo_noreturn(void)
{

}

extern "C" __declspec(noreturn) void __cdecl
__imp__invoke_watson(
    wchar_t const* const expression,
    wchar_t const* const function_name,
    wchar_t const* const file_name,
    unsigned int const line_number,
    uintptr_t const reserved)
{
	
}
#endif

namespace std {
	void __throw_bad_array_new_length() {
		ExitProcess(1);
	}

	void __throw_bad_cast() {
		ExitProcess(1);
	}

	void __throw_length_error(char const*) {
		ExitProcess(1);
	}

	void __throw_bad_alloc() {
		ExitProcess(1);
	}
}

void* operator new(std::size_t sz)
{
    if (sz == 0)
        ++sz;
 
    if (void *ptr = HeapAlloc(GetProcessHeap(), 0x0, sz))
        return ptr;
	else 
		return NULL;
    //throw std::bad_alloc{};
}
void operator delete(void* ptr) noexcept
{
    HeapFree(GetProcessHeap(), 0x0, ptr);
}

void operator delete(void* ptr, std::size_t sz)
{
	::operator delete(ptr);
}

void* memmove(void* dstpp, const void* srcpp, size_t len)
{
	char* dstp = (char*)dstpp;
	const char* srcp = (const char*)srcpp;

	if(dstp < srcp) {
		for (size_t i = 0; i < len; i++)
			*(dstp + i) = *(srcp + i);
	} else {
		for (size_t i = 0; i < len; i++)
			*(dstp + len - 1 - i) = *(srcp + len - 1 - i);
	}
	return dstpp;
}

void* memcpy(void* dstpp, const void* srcpp, size_t len)
{
	char* dstp = (char*)dstpp;
	const char* srcp = (const char*)srcpp;

	for (size_t i = 0; i < len; i++)
		*(dstp + i) = *(srcp + i);

	return dstpp;
}

#ifndef MSVC_BUILD
void* memset(void* s, int c,  size_t len)
{
	size_t i = 0;
    volatile unsigned char* p = (unsigned char*) s;
	while(i < len)
	{
		*p = c;
		p = p + 1;
		i = i + 1;
	}
    return s;
}
#endif

void runtimeMemset(char *p, char c, int l)
{
	#ifndef MSVC_BUILD
	for(int i = 0; i < l; i++)
		*(p + i) = '0';
	#endif
}

#ifndef MSVC_BUILD
inline int printf(const char* format, ...)
{
	va_list arg;
	int done;

	va_start(arg, format);
	done = ryuPrintfInternal(format, arg);
	va_end(arg);

	return done;
}
#endif

extern "C" {
	int _fltused = 0;
}

#endif