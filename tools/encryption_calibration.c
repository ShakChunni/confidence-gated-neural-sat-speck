#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#include "reference_encryption.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <sys/qos.h>
#include <pthread.h>
#endif

static int fail(const char *message) { fprintf(stderr, "%s\n", message); return 2; }
static int set_declared_policy(void) {
#ifdef __APPLE__
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) ||
        qos_class_self() != QOS_CLASS_USER_INITIATED) return 0;
    errno = 0;
    if (getpriority(PRIO_DARWIN_PROCESS, 0) != 0 || errno) return 0;
#endif
    errno = 0;
    int priority = getpriority(PRIO_PROCESS, 0);
    return !errno && priority == 0;
}
static double now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { perror("clock"); exit(2); }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static uint64_t digest(const reference_block *output, size_t n) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < n; ++i) {
        uint32_t packed = ((uint32_t)output[i].a << 16) | output[i].b;
        h = (h ^ packed) * UINT64_C(0x100000001b3);
    }
    return h;
}
static int integer(const char *s, uint64_t *result) {
    char *end;
    if (!s[0] || s[0] == '-') return 0;
    for (const char *p = s; *p; ++p) if (*p < '0' || *p > '9') return 0;
    errno = 0;
    *result = strtoull(s, &end, 10);
    return !errno && *end == '\0';
}
static int duration(const char *s, double *result) {
    char *end;
    errno = 0; *result = strtod(s, &end);
    return !errno && end != s && !*end && isfinite(*result) &&
           *result >= 0.001 && *result <= 10.0;
}
static int verify(void) {
    char line[512];
    unsigned records = 0;
    while (fgets(line, sizeof(line), stdin)) {
        if (!strchr(line, '\n') && !feof(stdin)) return fail("verification line too long");
        unsigned words[8];
        unsigned n = 0;
        for (char *token = strtok(line, " \t\r\n"); token; token = strtok(NULL, " \t\r\n")) {
            uint64_t value;
            if (n >= 8 || !integer(token, &value) || value > 65535)
                return fail("malformed verification input");
            words[n++] = (unsigned)value;
        }
        if (n != 8) return fail("malformed verification input");
        unsigned c = words[0], r = words[1];
        const unsigned *k = words + 2, *p = words + 6;
        if (c > 1 || r < 1 || r > (c ? 32U : 22U)) return fail("invalid profile");
        for (unsigned i = 0; i < 4; ++i) if (k[i] > 65535) return fail("invalid word");
        if (p[0] > 65535 || p[1] > 65535) return fail("invalid word");
        reference_key key = {{(uint16_t)k[0], (uint16_t)k[1], (uint16_t)k[2], (uint16_t)k[3]}};
        uint16_t rk[32];
        reference_expand(c, r, &key, rk);
        reference_block ct = reference_encrypt(c, r, (reference_block){p[0],p[1]}, rk);
        printf("{\"ciphertext\":[%u,%u],\"round_keys\":[", (unsigned)ct.a, (unsigned)ct.b);
        for (unsigned i = 0; i < r; ++i) printf("%s%u", i ? "," : "", (unsigned)rk[i]);
        puts("]}");
        ++records;
    }
    return records && !ferror(stdin) ? 0 : fail("empty or failed verification input");
}
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--verify")) return verify();
    if (argc == 2 && !strcmp(argv[1], "--policy-check")) {
        if (!set_declared_policy()) return fail("declared CPU policy unavailable");
#ifdef __APPLE__
        printf("{\"qos_class\":%u,\"nice\":0,\"threads\":1}\n", (unsigned)qos_class_self());
#else
        puts("{\"qos_class\":0,\"nice\":0,\"threads\":1}");
#endif
        return 0;
    }
    int measured = argc == 8 && !strcmp(argv[1], "--measure");
    if (!measured && !(argc == 6 && !strcmp(argv[1], "--batch")))
        return fail("--verify, --batch cipher rounds scope count, or --measure ... warmup seconds");
    unsigned c, charged;
    if (!strcmp(argv[2], "speck32_64")) c = 0;
    else if (!strcmp(argv[2], "simon32_64")) c = 1;
    else return fail("invalid cipher");
    if (!strcmp(argv[4], "expanded_key")) charged = 0;
    else if (!strcmp(argv[4], "schedule_charged")) charged = 1;
    else return fail("invalid scope");
    uint64_t r64, count64;
    if (!integer(argv[3], &r64) || r64 < 1 || r64 > (c ? 32U : 22U) ||
        !integer(argv[5], &count64) || count64 < 1 || count64 > 65536)
        return fail("invalid rounds or batch");
    unsigned rounds = (unsigned)r64;
    size_t count = (size_t)count64;
    double warmup = 0, minimum = 0;
    if (measured && (!duration(argv[6], &warmup) || !duration(argv[7], &minimum)))
        return fail("invalid duration");
    if (measured && !set_declared_policy()) return fail("declared CPU policy unavailable");
    reference_block *inputs = malloc(count * sizeof(*inputs));
    reference_block *outputs = malloc(count * sizeof(*outputs));
    reference_key *keys = malloc(count * sizeof(*keys));
    if (!inputs || !outputs || !keys) return fail("allocation failure");
    uint64_t state = UINT64_C(0xe5c0202610096002);
    for (size_t i = 0; i < count; ++i) {
        uint32_t p = (uint32_t)i * UINT32_C(0x9e3779b1) + UINT32_C(0x65656877);
        inputs[i] = (reference_block){(uint16_t)(p >> 16), (uint16_t)p};
        uint64_t k = splitmix64(&state);
        for (unsigned w = 0; w < 4; ++w) keys[i].words[w] = (uint16_t)(k >> (48 - 16*w));
    }
    reference_key fixed_key = {{0x1918, 0x1110, 0x0908, 0x0100}};
    uint16_t fixed_schedule[32];
    reference_expand(c, rounds, &fixed_key, fixed_schedule);
    uint64_t batches = 0;
    double elapsed = 0;
    if (measured) {
        double start = now();
        do { reference_batch(c, rounds, charged, count, inputs, keys, fixed_schedule, outputs);
        } while (now() - start < warmup);
        start = now();
        do {
            reference_batch(c, rounds, charged, count, inputs, keys, fixed_schedule, outputs);
            ++batches;
            elapsed = now() - start;
        } while (elapsed < minimum);
    } else {
        reference_batch(c, rounds, charged, count, inputs, keys, fixed_schedule, outputs);
        batches = 1;
    }
    unsigned qos = 0;
#ifdef __APPLE__
    qos = (unsigned)qos_class_self();
#endif
    errno = 0;
    int priority = getpriority(PRIO_PROCESS, 0);
    if (errno) priority = 999;
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage)) return fail("cannot read RSS");
#ifdef __APPLE__
    uint64_t rss = (uint64_t)usage.ru_maxrss;
#else
    uint64_t rss = (uint64_t)usage.ru_maxrss * 1024;
#endif
    printf("{\"cipher\":\"%s\",\"rounds\":%u,\"scope\":\"%s\","
           "\"batch_blocks\":%zu,\"batches\":%" PRIu64 ",\"blocks\":%" PRIu64 ","
           "\"elapsed_seconds\":%.9f,\"checksum\":\"%016" PRIx64 "\","
           "\"qos_class\":%u,\"nice\":%d,\"threads\":1,\"peak_rss_bytes\":%" PRIu64 "}\n",
           argv[2], rounds, argv[4], count, batches, batches*count, elapsed,
           digest(outputs, count), qos, priority, rss);
    free(inputs); free(outputs); free(keys);
    return 0;
}
