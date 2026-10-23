#ifndef SOLID_H
#define SOLID_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SOLID_API __declspec(dllimport)
#else
#define SOLID_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Input JSON contains "problem", "profile" and optional "inference_space".
 * AF/AAC require a cooperative profile. ANC may omit its recovery profile. */
SOLID_API char *solid_solve_json(const uint8_t *input, size_t input_len);
SOLID_API void solid_string_free(char *value);

#ifdef __cplusplus
}
#endif

#endif
