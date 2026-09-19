#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "SDL.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "doomtype.h"
#include "i_sr.h"
#include "m_argv.h"
#include "m_misc.h"

#define SR_REQUEST_MAGIC  0x31525341u
#define SR_RESPONSE_MAGIC 0x31525352u
#define SR_FORMAT_ARGB8888 1u
#define SR_HEADER_WORDS 7
#define SR_DEFAULT_SCALE "2"
#define SR_MAX_OUTPUT_BYTES (64u * 1024u * 1024u)

typedef struct
{
    boolean available;
    boolean attempted_init;
    byte *output_buffer;
    size_t output_buffer_size;
    char *worker_path;
    unsigned int last_frame_flags;

#ifdef _WIN32
    HANDLE process;
    HANDLE stdin_write;
    HANDLE stdout_read;
#else
    pid_t pid;
    int stdin_fd;
    int stdout_fd;
#endif
} sr_state_t;

static sr_state_t sr_state;

static const char *GetEnvOrDefault(const char *name, const char *default_value)
{
    const char *value;

    value = getenv(name);

    if (value == NULL || value[0] == '\0')
    {
        return default_value;
    }

    return value;
}

static boolean EnvIsFalse(const char *name)
{
    const char *value;

    value = getenv(name);

    return value != NULL
        && (strcmp(value, "0") == 0
         || strcasecmp(value, "false") == 0
         || strcasecmp(value, "off") == 0
         || strcasecmp(value, "no") == 0);
}

static char *TryWorkerPath(const char *base, const char *relative)
{
    char *path;

    if (base == NULL)
    {
        path = M_StringDuplicate(relative);
    }
    else
    {
        path = M_StringJoin(base, relative, NULL);
    }

    if (M_FileExists(path))
    {
        return path;
    }

    free(path);
    return NULL;
}

static char *FindWorkerPath(void)
{
    static const char *relative_paths[] = {
        "SR" DIR_SEPARATOR_S "sr_worker.py",
        ".." DIR_SEPARATOR_S "SR" DIR_SEPARATOR_S "sr_worker.py",
        ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S "SR" DIR_SEPARATOR_S "sr_worker.py",
        ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S "SR" DIR_SEPARATOR_S "sr_worker.py",
        ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S "SR" DIR_SEPARATOR_S "sr_worker.py",
        ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S ".." DIR_SEPARATOR_S "SR" DIR_SEPARATOR_S "sr_worker.py",
    };
    const char *env_path;
    char *base_path;
    char *result;
    unsigned int i;

    env_path = getenv("AVION_SR_WORKER");

    if (env_path != NULL && env_path[0] != '\0')
    {
        if (M_FileExists(env_path))
        {
            return M_StringDuplicate(env_path);
        }

        fprintf(stderr, "SR: AVION_SR_WORKER does not exist: %s\n", env_path);
        return NULL;
    }

    for (i = 0; i < arrlen(relative_paths); ++i)
    {
        result = TryWorkerPath(NULL, relative_paths[i]);

        if (result != NULL)
        {
            return result;
        }
    }

    base_path = SDL_GetBasePath();

    if (base_path != NULL)
    {
        for (i = 0; i < arrlen(relative_paths); ++i)
        {
            result = TryWorkerPath(base_path, relative_paths[i]);

            if (result != NULL)
            {
                SDL_free(base_path);
                return result;
            }
        }

        SDL_free(base_path);
    }

    return NULL;
}

static boolean EnsureOutputBuffer(size_t size)
{
    byte *new_buffer;

    if (size <= sr_state.output_buffer_size)
    {
        return true;
    }

    new_buffer = realloc(sr_state.output_buffer, size);

    if (new_buffer == NULL)
    {
        fprintf(stderr, "SR: failed to allocate %lu output bytes\n",
                (unsigned long) size);
        return false;
    }

    sr_state.output_buffer = new_buffer;
    sr_state.output_buffer_size = size;

    return true;
}

#ifdef _WIN32

static char *QuoteWindowsArgument(const char *arg)
{
    char *quoted;
    char *out;
    const char *in;
    size_t len;

    len = 3;

    for (in = arg; *in != '\0'; ++in)
    {
        len += *in == '"' ? 2 : 1;
    }

    quoted = malloc(len);

    if (quoted == NULL)
    {
        return NULL;
    }

    out = quoted;
    *out++ = '"';

    for (in = arg; *in != '\0'; ++in)
    {
        if (*in == '"')
        {
            *out++ = '\\';
        }

        *out++ = *in;
    }

    *out++ = '"';
    *out = '\0';

    return quoted;
}

static char *BuildCommandLine(const char *python,
                              const char *worker,
                              const char *scale,
                              const char *checkpoint)
{
    char *python_arg;
    char *worker_arg;
    char *checkpoint_arg;
    char *command;

    python_arg = QuoteWindowsArgument(python);
    worker_arg = QuoteWindowsArgument(worker);

    if (python_arg == NULL || worker_arg == NULL)
    {
        free(python_arg);
        free(worker_arg);
        return NULL;
    }

    if (checkpoint != NULL && checkpoint[0] != '\0')
    {
        checkpoint_arg = QuoteWindowsArgument(checkpoint);

        if (checkpoint_arg == NULL)
        {
            free(python_arg);
            free(worker_arg);
            return NULL;
        }

        command = M_StringJoin(python_arg, " -u ", worker_arg,
                               " --scale ", scale,
                               " --checkpoint ", checkpoint_arg,
                               NULL);
        free(checkpoint_arg);
    }
    else
    {
        command = M_StringJoin(python_arg, " -u ", worker_arg,
                               " --scale ", scale,
                               NULL);
    }

    free(python_arg);
    free(worker_arg);

    return command;
}

static boolean SpawnWorker(const char *python,
                           const char *worker,
                           const char *scale,
                           const char *checkpoint)
{
    SECURITY_ATTRIBUTES sa;
    STARTUPINFOA startup_info;
    PROCESS_INFORMATION process_info;
    HANDLE child_stdin_read;
    HANDLE child_stdout_write;
    HANDLE stderr_handle;
    char *command_line;
    BOOL ok;

    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    child_stdin_read = NULL;
    child_stdout_write = NULL;
    stderr_handle = NULL;

    if (!CreatePipe(&child_stdin_read, &sr_state.stdin_write, &sa, 0))
    {
        return false;
    }

    if (!SetHandleInformation(sr_state.stdin_write, HANDLE_FLAG_INHERIT, 0))
    {
        return false;
    }

    if (!CreatePipe(&sr_state.stdout_read, &child_stdout_write, &sa, 0))
    {
        return false;
    }

    if (!SetHandleInformation(sr_state.stdout_read, HANDLE_FLAG_INHERIT, 0))
    {
        return false;
    }

    stderr_handle = CreateFileA("NUL", GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE,
                                &sa, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, NULL);

    if (stderr_handle == INVALID_HANDLE_VALUE)
    {
        stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
    }

    command_line = BuildCommandLine(python, worker, scale, checkpoint);

    if (command_line == NULL)
    {
        return false;
    }

    memset(&startup_info, 0, sizeof(startup_info));
    memset(&process_info, 0, sizeof(process_info));
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdInput = child_stdin_read;
    startup_info.hStdOutput = child_stdout_write;
    startup_info.hStdError = stderr_handle;

    ok = CreateProcessA(NULL, command_line, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &startup_info, &process_info);

    free(command_line);
    CloseHandle(child_stdin_read);
    CloseHandle(child_stdout_write);

    if (stderr_handle != GetStdHandle(STD_ERROR_HANDLE)
     && stderr_handle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(stderr_handle);
    }

    if (!ok)
    {
        CloseHandle(sr_state.stdin_write);
        CloseHandle(sr_state.stdout_read);
        sr_state.stdin_write = NULL;
        sr_state.stdout_read = NULL;
        return false;
    }

    sr_state.process = process_info.hProcess;
    CloseHandle(process_info.hThread);
    return true;
}

static boolean WriteExact(const void *buffer, size_t size)
{
    const byte *cursor;
    DWORD written;

    cursor = buffer;

    while (size > 0)
    {
        written = 0;

        if (!WriteFile(sr_state.stdin_write, cursor,
                       size > 0x7fffffff ? 0x7fffffff : (DWORD) size,
                       &written, NULL)
         || written == 0)
        {
            return false;
        }

        cursor += written;
        size -= written;
    }

    return true;
}

static boolean ReadExact(void *buffer, size_t size)
{
    byte *cursor;
    DWORD bytes_read;

    cursor = buffer;

    while (size > 0)
    {
        bytes_read = 0;

        if (!ReadFile(sr_state.stdout_read, cursor,
                      size > 0x7fffffff ? 0x7fffffff : (DWORD) size,
                      &bytes_read, NULL)
         || bytes_read == 0)
        {
            return false;
        }

        cursor += bytes_read;
        size -= bytes_read;
    }

    return true;
}

#else

static boolean SpawnWorker(const char *python,
                           const char *worker,
                           const char *scale,
                           const char *checkpoint)
{
    int stdin_pipe[2];
    int stdout_pipe[2];

    if (pipe(stdin_pipe) != 0)
    {
        return false;
    }

    if (pipe(stdout_pipe) != 0)
    {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return false;
    }

    sr_state.pid = fork();

    if (sr_state.pid < 0)
    {
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return false;
    }

    if (sr_state.pid == 0)
    {
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        if (checkpoint != NULL && checkpoint[0] != '\0')
        {
            execlp(python, python, "-u", worker, "--scale", scale,
                   "--checkpoint", checkpoint, (char *) NULL);
        }
        else
        {
            execlp(python, python, "-u", worker, "--scale", scale,
                   (char *) NULL);
        }

        _exit(127);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    sr_state.stdin_fd = stdin_pipe[1];
    sr_state.stdout_fd = stdout_pipe[0];

    return true;
}

static boolean WriteExact(const void *buffer, size_t size)
{
    const byte *cursor;
    ssize_t written;

    cursor = buffer;

    while (size > 0)
    {
        written = write(sr_state.stdin_fd, cursor, size);

        if (written <= 0)
        {
            return false;
        }

        cursor += written;
        size -= written;
    }

    return true;
}

static boolean ReadExact(void *buffer, size_t size)
{
    byte *cursor;
    ssize_t bytes_read;

    cursor = buffer;

    while (size > 0)
    {
        bytes_read = read(sr_state.stdout_fd, cursor, size);

        if (bytes_read <= 0)
        {
            return false;
        }

        cursor += bytes_read;
        size -= bytes_read;
    }

    return true;
}

#endif

void I_SR_Shutdown(void)
{
#ifdef _WIN32
    if (sr_state.stdin_write != NULL)
    {
        CloseHandle(sr_state.stdin_write);
        sr_state.stdin_write = NULL;
    }

    if (sr_state.stdout_read != NULL)
    {
        CloseHandle(sr_state.stdout_read);
        sr_state.stdout_read = NULL;
    }

    if (sr_state.process != NULL)
    {
        if (WaitForSingleObject(sr_state.process, 1000) == WAIT_TIMEOUT)
        {
            TerminateProcess(sr_state.process, 0);
        }

        CloseHandle(sr_state.process);
        sr_state.process = NULL;
    }
#else
    if (sr_state.stdin_fd >= 0)
    {
        close(sr_state.stdin_fd);
        sr_state.stdin_fd = -1;
    }

    if (sr_state.stdout_fd >= 0)
    {
        close(sr_state.stdout_fd);
        sr_state.stdout_fd = -1;
    }

    if (sr_state.pid > 0)
    {
        if (waitpid(sr_state.pid, NULL, WNOHANG) == 0)
        {
            kill(sr_state.pid, SIGTERM);
            waitpid(sr_state.pid, NULL, 0);
        }

        sr_state.pid = 0;
    }
#endif

    free(sr_state.output_buffer);
    sr_state.output_buffer = NULL;
    sr_state.output_buffer_size = 0;

    free(sr_state.worker_path);
    sr_state.worker_path = NULL;

    sr_state.available = false;
    sr_state.last_frame_flags = 0;
}

void I_SR_Init(void)
{
    const char *python;
    const char *scale;
    const char *checkpoint;

    if (sr_state.attempted_init)
    {
        return;
    }

    sr_state.attempted_init = true;

#ifndef _WIN32
    sr_state.stdin_fd = -1;
    sr_state.stdout_fd = -1;
#endif

    if (M_ParmExists("-nosr") || EnvIsFalse("AVION_SR"))
    {
        fprintf(stderr, "SR: disabled\n");
        return;
    }

    sr_state.worker_path = FindWorkerPath();

    if (sr_state.worker_path == NULL)
    {
        fprintf(stderr, "SR: worker script was not found; using normal renderer\n");
        return;
    }

    python = GetEnvOrDefault("AVION_SR_PYTHON", "python");
    scale = GetEnvOrDefault("AVION_SR_SCALE", SR_DEFAULT_SCALE);
    checkpoint = getenv("AVION_SR_CHECKPOINT");

    if (!SpawnWorker(python, sr_state.worker_path, scale, checkpoint))
    {
        fprintf(stderr, "SR: failed to start worker; using normal renderer\n");
        I_SR_Shutdown();
        return;
    }

    sr_state.available = true;
    fprintf(stderr, "SR: worker started: %s\n", sr_state.worker_path);
}

boolean I_SR_IsAvailable(void)
{
    return sr_state.available;
}

unsigned int I_SR_GetLastFrameFlags(void)
{
    return sr_state.last_frame_flags;
}

boolean I_SR_ProcessFrame(const void *pixels,
                          int width,
                          int height,
                          int pitch,
                          const byte **output_pixels,
                          int *output_width,
                          int *output_height,
                          int *output_pitch)
{
    uint32_t request_header[SR_HEADER_WORDS];
    uint32_t response_header[SR_HEADER_WORDS];
    size_t input_bytes;
    size_t output_bytes;

    if (!sr_state.available)
    {
        sr_state.last_frame_flags = 0;
        return false;
    }

    if (pixels == NULL || width <= 0 || height <= 0 || pitch < width * 4)
    {
        sr_state.last_frame_flags = 0;
        return false;
    }

    input_bytes = (size_t) pitch * (size_t) height;

    request_header[0] = SR_REQUEST_MAGIC;
    request_header[1] = (uint32_t) width;
    request_header[2] = (uint32_t) height;
    request_header[3] = (uint32_t) pitch;
    request_header[4] = SR_FORMAT_ARGB8888;
    request_header[5] = 0;
    request_header[6] = (uint32_t) input_bytes;

    if (!WriteExact(request_header, sizeof(request_header))
     || !WriteExact(pixels, input_bytes)
     || !ReadExact(response_header, sizeof(response_header)))
    {
        fprintf(stderr, "SR: worker communication failed; disabling SR\n");
        I_SR_Shutdown();
        return false;
    }

    output_bytes = response_header[6];

    if (response_header[0] != SR_RESPONSE_MAGIC
     || response_header[4] != SR_FORMAT_ARGB8888
     || response_header[1] == 0
     || response_header[2] == 0
     || response_header[3] < response_header[1] * 4
     || output_bytes == 0
     || output_bytes > SR_MAX_OUTPUT_BYTES)
    {
        fprintf(stderr, "SR: invalid worker response; disabling SR\n");
        I_SR_Shutdown();
        return false;
    }

    if (!EnsureOutputBuffer(output_bytes)
     || !ReadExact(sr_state.output_buffer, output_bytes))
    {
        fprintf(stderr, "SR: failed to read worker output; disabling SR\n");
        I_SR_Shutdown();
        return false;
    }

    *output_pixels = sr_state.output_buffer;
    *output_width = (int) response_header[1];
    *output_height = (int) response_header[2];
    *output_pitch = (int) response_header[3];
    sr_state.last_frame_flags = response_header[5];

    return true;
}
