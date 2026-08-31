// Science-grade windowed Wang-Landau + 1/t refinement + Metropolis-Hastings
// Monte Carlo estimator for the number of 4-dimensional partitions.
//
//
// AUTHOR: Avinandan Mondal.
// Email: avinandan.physics@proton.me
//
// -----------------------------------------------------------------------------------
// Copyright (C) 2026 Avinandan Mondal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, in particular version 3 of the
// License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: AGPL-3.0-only
//
//
// -------------------------------------------------------------------------------------
// The algorithm is tuned for science-grade large-N runs and conservative WL convergence:
//   * the level range needed for the forward broad-histogram recursion is split
//     into overlapping windows;
//   * each window first completes repeated stringent conventional-WL flatness stages;
//   * only then does it enter a Belardinelli-Pereyra-style 1/t WL refinement walk;
//   * the frozen local DOS weight is then used for a stationary MH production run;
//   * windows are independent and therefore run in parallel with OpenMP;
//   * production samples from overlapping windows are pooled level-by-level;
//   * Delta_4 = log p_4(n)-log p_4(n-1) is computed and stored in DELTA4_CSV_FILE;
//   * only final p_4(N) is reported in the terminal console;
//   * RNG_SEED stored in SEED_FILE if set at 0ULL (runtime root seed generation).
//
//
//
//
// Compile with OpenMP, link math library and use appropriate optimization flags.

//
// -----------------------------------------------------------------------------
// User-facing macros
// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <inttypes.h>
#include <omp.h>
#include <unistd.h>

#define DELTA4_CSV_FILE "MC4D_Delta4.csv"
#define SEED_FILE "MC4D_seed.txt"

#ifndef N
#define N 60
#endif

#ifndef KNOWN_N
#define KNOWN_N 40
#endif

#ifndef KNOWN_P4
#define KNOWN_P4 118137556912895.0L
#endif

// Default thread count. Change this macro or pass e.g. -DMC_THREADS=8.
#ifndef MC_THREADS
#define MC_THREADS 4
#endif

// Number of completely independent Monte Carlo runs whose production
// sufficient statistics are pooled.
#ifndef NUM_RUNS
#define NUM_RUNS 1
#endif

// ------------------------ Window parameters ---------------------------------
// Window geometry is computed once by the preprocessor from N and KNOWN_N.
// This creates several independent tasks even for short validation runs, while
// recovering the original 128/32 geometry at large N.
#define WL_REQUIRED_LEVEL_SPAN (N - KNOWN_N + 1)

#if WL_REQUIRED_LEVEL_SPAN <= 16
# define WL_WINDOW_WIDTH    4
# define WL_WINDOW_OVERLAP  2
#elif WL_REQUIRED_LEVEL_SPAN <= 64
# define WL_WINDOW_WIDTH    8
# define WL_WINDOW_OVERLAP  3
#elif WL_REQUIRED_LEVEL_SPAN <= 256
# define WL_WINDOW_WIDTH   16
# define WL_WINDOW_OVERLAP  4
#elif WL_REQUIRED_LEVEL_SPAN <= 1024
# define WL_WINDOW_WIDTH   32
# define WL_WINDOW_OVERLAP  8
#elif WL_REQUIRED_LEVEL_SPAN <= 4096
# define WL_WINDOW_WIDTH   64
# define WL_WINDOW_OVERLAP 16
#else
# define WL_WINDOW_WIDTH  128
# define WL_WINDOW_OVERLAP 32
#endif

// For large science runs, move the lower edge of the first window below the
// anchor so KNOWN_N is comfortably inside the first DOS window. The level
// cannot be negative, so the start is clamped at zero.
#ifndef WL_FIRST_WINDOW_PAD_THRESHOLD
#define WL_FIRST_WINDOW_PAD_THRESHOLD 256
#endif

// The reference work width follows the same hierarchy.  Small windows get
// enough steps to remain statistically reliable, while N>=1024 recovers the
// original 128-level budget per window.
#if WL_REQUIRED_LEVEL_SPAN <= 16
# define WL_STEP_REFERENCE_WIDTH 16ULL
#elif WL_REQUIRED_LEVEL_SPAN <= 64
# define WL_STEP_REFERENCE_WIDTH 32ULL
#elif WL_REQUIRED_LEVEL_SPAN <= 256
# define WL_STEP_REFERENCE_WIDTH 64ULL
#else
# define WL_STEP_REFERENCE_WIDTH 128ULL
#endif

// ------------------------ 1/t WL parameters ---------------------------------
// Initial logarithmic modification factor F = ln(f).
#ifndef WL_INITIAL_LNF
#define WL_INITIAL_LNF 1.0L
#endif

// First-phase WL uses histogram flatness until F <= n_levels/t; then the
// Belardinelli-Pereyra 1/t regime is entered, with F(t)=WL_1T_SCALE*n_levels/t.
#ifndef WL_FLATNESS
#define WL_FLATNESS 0.95L
#endif
#ifndef WL_CHECK_INTERVAL
#define WL_CHECK_INTERVAL 10000ULL
#endif
#ifndef WL_MIN_FLAT_STAGES
#define WL_MIN_FLAT_STAGES 16
#endif
#ifndef WL_MIN_VISITS_PER_BIN
#define WL_MIN_VISITS_PER_BIN 3000ULL
#endif
#ifndef WL_1T_SCALE
#define WL_1T_SCALE 1.0L
#endif

// Number of steps in the genuine 1/t phase in EACH window. This is the main
// accuracy/performance knob for large-N production work.
#ifndef WL_1T_STEPS
#define WL_1T_STEPS 1000000ULL
#endif

// Safety limit on total WL steps in one window.
#ifndef WL_MAX_STEPS
#define WL_MAX_STEPS 5000000000ULL
#endif

// ------------------------ Production parameters -----------------------------
#ifndef PROD_BURN_IN
#define PROD_BURN_IN 5000ULL
#endif
#ifndef PROD_STEPS
#define PROD_STEPS 1000000ULL
#endif

// RNG_SEED == 0 means runtime seeding. Any nonzero value is reproducible.
#ifndef RNG_SEED
#define RNG_SEED 0ULL
#endif

// Increase to 1 for expensive full-state consistency checks.
#ifndef CHECK_STATE
#define CHECK_STATE 0
#endif
#ifndef CHECK_INTERVAL
#define CHECK_INTERVAL 1000000ULL
#endif
#ifndef VERBOSE_PROGRESS
#define VERBOSE_PROGRESS 1
#endif

// Hash-table load factor.
#ifndef HASH_LOAD_NUM
#define HASH_LOAD_NUM 1
#endif
#ifndef HASH_LOAD_DEN
#define HASH_LOAD_DEN 2
#endif

// -----------------------------------------------------------------------------
// Basic checks
// -----------------------------------------------------------------------------

#if (KNOWN_N < 1)
#error "KNOWN_N must be >= 1"
#endif
#if (KNOWN_N > N)
#error "KNOWN_N must be <= N"
#endif
#if (MC_THREADS < 1)
#error "MC_THREADS must be >= 1"
#endif
#if (WL_WINDOW_WIDTH < 2)
#error "WL_WINDOW_WIDTH must be >= 2"
#endif
#if (WL_WINDOW_OVERLAP < 1) || (WL_WINDOW_OVERLAP >= WL_WINDOW_WIDTH)
#error "WL_WINDOW_OVERLAP must satisfy 1 <= overlap < width"
#endif
#if (HASH_LOAD_NUM <= 0) || (HASH_LOAD_DEN <= 0) || (HASH_LOAD_NUM >= HASH_LOAD_DEN)
#error "Invalid hash load-factor macros"
#endif

// -----------------------------------------------------------------------------
// Five-dimensional lattice nodes and open-addressing node sets
// -----------------------------------------------------------------------------

typedef struct {
    __uint128_t key;              // 5 packed coordinates, d=0 in low bits.
} Node;

typedef struct {
    Node *items;               // Same insertion/removal order as the reference.
    uint64_t *item_hash;       // Cached hash for every occupied item.
    int count;
    int capacity;
    // 0 = empty, >0 = item index + 1.  No tombstones: deletion uses
    // backward-shift repair of the open-addressing cluster.
    int32_t *slot;
    int32_t *item_slot;
    int hcap;
} NodeSet;

/* Coordinate packing is selected once in main().  For N ~= 15000--20000 this is 15 bits/coordinate, i.e. 75 bits total. */
static unsigned g_coord_bits = 0;
static __uint128_t g_coord_mask = 0;
static __uint128_t g_coord_step[5] = {0, 0, 0, 0, 0};
static long double *g_log_inv_count = NULL;
static long double *g_log_half_over_count = NULL;
static size_t g_log_count_cap = 0;

static inline __uint128_t coord_mask_for_dim(int d)
{
    return g_coord_mask << (d * g_coord_bits);
}

static inline bool node_coord_is_zero(const Node *n, int d)
{
    return (n->key & coord_mask_for_dim(d)) == 0;
}

static inline Node node_axis(int d, unsigned value)
{
    Node n = {0};
    n.key = (__uint128_t)value << (d * g_coord_bits);
    return n;
}

static inline Node node_plus(const Node *n, int d)
{
    Node r = *n;
    r.key += g_coord_step[d];
    return r;
}

static inline Node node_minus(const Node *n, int d)
{
    Node r = *n;
    r.key -= g_coord_step[d];
    return r;
}

typedef struct {
    uint64_t state[4];
} RNG;

typedef struct {
    int level;              // number of boxes = level + 1
    NodeSet part_set;       // occupied lattice points
    NodeSet growth;         // distinct legal additions
    NodeSet remove;         // distinct removable nodes
} ChainState;

typedef struct {
    int lo;                 // inclusive level
    int hi;                 // inclusive level
    int len;                // hi-lo+1
} Window;

typedef struct {
    uint64_t *hits;         // len entries
    long double *gp_sum;    // len entries
    long double *gm_sum;    // len entries
    uint64_t wl_steps;
    uint64_t production_steps;
    uint64_t accepted;
    uint64_t attempted;
    long double final_lnf;
} WindowResult;

static void die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
}

static inline bool node_equal(const Node *a, const Node *b)
{
    return a->key == b->key;
}

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t node_hash(const Node *a)
{
    /* Fold all 75 packed coordinate bits into one 64-bit hash. */
    const uint64_t lo = (uint64_t)a->key;
    const uint64_t hi = (uint64_t)(a->key >> 64);
    uint64_t h = lo ^ ((hi << 29) | (hi >> 35));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static int next_power_of_two(int x)
{
    int p = 16;
    while (p < x) {
        if (p > INT32_MAX / 2) die("Hash table size overflow");
        p <<= 1;
    }
    return p;
}

static void nodeset_rebuild(NodeSet *s, int requested_hcap)
{
    int hcap = next_power_of_two(requested_hcap);
    int32_t *newslot = (int32_t *)calloc((size_t)hcap, sizeof(int32_t));
    if (!newslot) die("Out of memory in hash-table rebuild");

    for (int idx = 0; idx < s->count; ++idx) {
        const uint64_t h = s->item_hash[idx];
        int pos = (int)(h & (uint64_t)(hcap - 1));
        while (newslot[pos] != 0)
            pos = (pos + 1) & (hcap - 1);
        newslot[pos] = (int32_t)(idx + 1);
        s->item_slot[idx] = (int32_t)pos;
    }

    free(s->slot);
    s->slot = newslot;
    s->hcap = hcap;
}

static void nodeset_init(NodeSet *s, int capacity)
{
    memset(s, 0, sizeof *s);
    s->capacity = capacity;
    s->items = (Node *)malloc((size_t)capacity * sizeof(Node));
    s->item_hash = (uint64_t *)malloc((size_t)capacity * sizeof(uint64_t));
    if (!s->items || !s->item_hash) die("Out of memory allocating node set");
    s->hcap = next_power_of_two((capacity * HASH_LOAD_DEN) / HASH_LOAD_NUM + 16);
    s->slot = (int32_t *)calloc((size_t)s->hcap, sizeof(int32_t));
    s->item_slot = (int32_t *)malloc((size_t)capacity * sizeof(int32_t));
    if (!s->slot || !s->item_slot) die("Out of memory allocating hash table");
}

static void nodeset_free(NodeSet *s)
{
    free(s->items);
    free(s->item_hash);
    free(s->slot);
    free(s->item_slot);
    memset(s, 0, sizeof *s);
}

static inline int nodeset_find_slot_h(const NodeSet *s, const Node *node, uint64_t h)
{
    int pos = (int)(h & (uint64_t)(s->hcap - 1));
    for (;;) {
        const int32_t v = s->slot[pos];
        if (v == 0) return -1;
        const int idx = v - 1;
        if (s->item_hash[idx] == h && node_equal(&s->items[idx], node))
            return pos;
        pos = (pos + 1) & (s->hcap - 1);
    }
}

static inline int nodeset_find_slot(const NodeSet *s, const Node *node)
{
    return nodeset_find_slot_h(s, node, node_hash(node));
}

static inline bool nodeset_contains(const NodeSet *s, const Node *node)
{
    return nodeset_find_slot(s, node) >= 0;
}

static inline bool nodeset_add(NodeSet *s, const Node *node)
{
    if (s->count >= s->capacity) die("NodeSet capacity exceeded");

    const uint64_t h = node_hash(node);
    int pos = (int)(h & (uint64_t)(s->hcap - 1));
    for (;;) {
        const int32_t v = s->slot[pos];
        if (v == 0) break;
        const int idx = v - 1;
        if (s->item_hash[idx] == h && node_equal(&s->items[idx], node))
            return false;
        pos = (pos + 1) & (s->hcap - 1);
    }

    if ((s->count + 1) * HASH_LOAD_DEN >=
        s->hcap * HASH_LOAD_NUM) {
        nodeset_rebuild(s, s->hcap * 2);
        pos = (int)(h & (uint64_t)(s->hcap - 1));
        while (s->slot[pos] != 0)
            pos = (pos + 1) & (s->hcap - 1);
    }

    const int idx = s->count;
    s->items[idx] = *node;
    s->item_hash[idx] = h;
    s->slot[pos] = (int32_t)(idx + 1);
    s->item_slot[idx] = (int32_t)pos;
    s->count = idx + 1;
    return true;
}

static inline void nodeset_add_absent(NodeSet *s, const Node *node)
{
    if (s->count >= s->capacity) die("NodeSet capacity exceeded");
    const uint64_t h = node_hash(node);
    int pos = (int)(h & (uint64_t)(s->hcap - 1));
    while (s->slot[pos] != 0) pos = (pos + 1) & (s->hcap - 1);
    const int idx = s->count++;
    s->items[idx] = *node;
    s->item_hash[idx] = h;
    s->slot[pos] = (int32_t)(idx + 1);
    s->item_slot[idx] = (int32_t)pos;
}

static inline void nodeset_add_absent_h(NodeSet *s, const Node *node, uint64_t h)
{
    if (s->count >= s->capacity) die("NodeSet capacity exceeded");
    int pos = (int)(h & (uint64_t)(s->hcap - 1));
    while (s->slot[pos] != 0) pos = (pos + 1) & (s->hcap - 1);
    const int idx = s->count++;
    s->items[idx] = *node;
    s->item_hash[idx] = h;
    s->slot[pos] = (int32_t)(idx + 1);
    s->item_slot[idx] = (int32_t)pos;
}

static inline void nodeset_shift_delete_from(NodeSet *s, int hole)
{
    const int mask = s->hcap - 1;
    int pos = (hole + 1) & mask;

    for (;;) {
        const int32_t v = s->slot[pos];
        if (v == 0) return;

        const int idx = v - 1;
        const uint64_t h = s->item_hash[idx];
        const int home = (int)(h & (uint64_t)mask);

        /* Move an entry back only when its probe interval crosses the hole. */
        const unsigned dist_to_pos = (unsigned)(pos - home) & (unsigned)mask;
        const unsigned dist_to_hole = (unsigned)(pos - hole) & (unsigned)mask;
        const bool crosses = dist_to_pos >= dist_to_hole;
        if (crosses) {
            s->slot[hole] = v;
            s->item_slot[idx] = (int32_t)hole;
            s->slot[pos] = 0;
            hole = pos;
        }
        pos = (pos + 1) & mask;
    }
}

static inline bool nodeset_remove_slot(NodeSet *s, int pos)
{
    const int32_t v = s->slot[pos];
    if (v <= 0) return false;

    const int idx = v - 1;
    const int last = s->count - 1;

    if (idx != last) {
        const Node moved = s->items[last];
        const uint64_t moved_hash = s->item_hash[last];
        const int moved_slot = s->item_slot[last];
        s->items[idx] = moved;
        s->item_hash[idx] = moved_hash;
        s->item_slot[idx] = moved_slot;
        s->slot[moved_slot] = (int32_t)(idx + 1);
    }

    --s->count;
    s->slot[pos] = 0;
    nodeset_shift_delete_from(s, pos);
    return true;
}

static inline bool nodeset_remove_h(NodeSet *s, const Node *node, uint64_t h)
{
    const int pos = nodeset_find_slot_h(s, node, h);
    if (pos < 0) return false;
    return nodeset_remove_slot(s, pos);
}

static inline bool nodeset_remove(NodeSet *s, const Node *node)
{
    const int pos = nodeset_find_slot(s, node);
    if (pos < 0) return false;
    return nodeset_remove_slot(s, pos);
}

#if CHECK_STATE
static void nodeset_clear(NodeSet *s)
{
    s->count = 0;
    memset(s->slot, 0, (size_t)s->hcap * sizeof(int32_t));
}
#endif

// -----------------------------------------------------------------------------
// Per-chain RNG. No shared mutable RNG state => negligible thread overhead.
// -----------------------------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256ss(RNG *r)
{
    const uint64_t result = rotl64(r->state[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = r->state[1] << 17;
    r->state[2] ^= r->state[0];
    r->state[3] ^= r->state[1];
    r->state[1] ^= r->state[2];
    r->state[0] ^= r->state[3];
    r->state[2] ^= t;
    r->state[3] = rotl64(r->state[3], 45);
    return result;
}

static uint64_t runtime_seed(void)
{
    struct timespec ts;
    uint64_t seed = 0x6a09e667f3bcc909ULL;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        seed ^= (uint64_t)ts.tv_sec * 0x9e3779b97f4a7c15ULL;
        seed ^= (uint64_t)ts.tv_nsec * 0xbf58476d1ce4e5b9ULL;
    } else {
        seed ^= (uint64_t)time(NULL) * 0x9e3779b97f4a7c15ULL;
        seed ^= (uint64_t)clock() * 0xbf58476d1ce4e5b9ULL;
    }
    seed ^= (uint64_t)(uintptr_t)&ts * 0x94d049bb133111ebULL;
    seed ^= (uint64_t)getpid() * 0x517cc1b727220a95ULL;
    return splitmix64(seed);
}

static void seed_rng(RNG *r, uint64_t seed)
{
    uint64_t x = seed;
    for (int i = 0; i < 4; ++i) {
        x += 0x9e3779b97f4a7c15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->state[i] = z ^ (z >> 31);
    }
    if ((r->state[0] | r->state[1] | r->state[2] | r->state[3]) == 0) {
        die("RNG was seeded to zero state");
    }
}

static inline uint32_t rand_index(RNG *r, uint32_t m)
{
    if (m == 0) die("rand_index(0)");
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % m);
    uint64_t x;
    do {
        x = xoshiro256ss(r);
    } while (x >= limit);
    return (uint32_t)(x % m);
}

static inline long double rand_unit(RNG *r)
{
    return (long double)(xoshiro256ss(r) >> 11) *
           (1.0L / 9007199254740992.0L);
}

static inline bool rand_bit_half(RNG *r)
{
    return xoshiro256ss(r) < UINT64_C(0x8000000000000000);
}

static inline long double rand_log_unit(RNG *r)
{
    long double u;
    do {
        u = rand_unit(r);
    } while (u == 0.0L);
    return logl(u);
}

// -----------------------------------------------------------------------------
// Partition graph helpers
// -----------------------------------------------------------------------------

static inline bool part_contains(const ChainState *s, const Node *node)
{
    return nodeset_contains(&s->part_set, node);
}

static bool is_growth_in_set(const NodeSet *part_set, const Node *node)
{
    for (int i = 0; i < 5; ++i) {
        if (node_coord_is_zero(node, i)) continue;
        Node pred = node_minus(node, i);
        if (!nodeset_contains(part_set, &pred)) return false;
    }
    return true;
}

static bool is_removable_in_set(const NodeSet *part_set, const Node *node)
{
    for (int i = 0; i < 5; ++i) {
        Node succ = node_plus(node, i);
        if (nodeset_contains(part_set, &succ)) return false;
    }
    return true;
}

static void build_initial_state(ChainState *s)
{
    memset(s, 0, sizeof *s);
    s->level = 0;
    nodeset_init(&s->part_set, N + 2);
    nodeset_init(&s->growth, 5 * N + 8);
    nodeset_init(&s->remove, N + 2);

    Node origin = {0};
    if (!nodeset_add(&s->part_set, &origin)) die("Failed to initialize partition");
    if (!nodeset_add(&s->remove, &origin)) die("Failed to initialize remove set");

    for (int j = 0; j < 5; ++j) {
        Node g = node_axis(j, 1);
        if (!nodeset_add(&s->growth, &g)) die("Failed to initialize growth set");
    }
}

static void free_state(ChainState *s)
{
    nodeset_free(&s->part_set);
    nodeset_free(&s->growth);
    nodeset_free(&s->remove);
}

#if CHECK_STATE
static void consistency_check(const ChainState *s)
{
    NodeSet gcheck, rcheck;
    nodeset_init(&gcheck, 5 * N + 8);
    nodeset_init(&rcheck, N + 2);
    nodeset_clear(&gcheck);
    nodeset_clear(&rcheck);

    for (int i = 0; i < s->part_set.count; ++i) {
        const Node *p = &s->part_set.items[i];
        if (is_removable_in_set(&s->part_set, p)) nodeset_add(&rcheck, p);
        for (int j = 0; j < 5; ++j) {
            Node cand = *p;
            cand = node_plus(&cand, j);
            if (!part_contains(s, &cand) && is_growth_in_set(&s->part_set, &cand)) {
                nodeset_add(&gcheck, &cand);
            }
        }
    }

    if (gcheck.count != s->growth.count || rcheck.count != s->remove.count) {
        die("Incremental graph-state degree mismatch");
    }
    for (int i = 0; i < gcheck.count; ++i)
        if (!nodeset_contains(&s->growth, &gcheck.items[i])) die("Growth-set mismatch");
    for (int i = 0; i < rcheck.count; ++i)
        if (!nodeset_contains(&s->remove, &rcheck.items[i])) die("Remove-set mismatch");

    nodeset_free(&gcheck);
    nodeset_free(&rcheck);
}
#endif

static inline void predict_add(const ChainState *s, const Node *q,
                                  int *new_gp, int *new_gm,
                                  uint8_t *lost_remove_mask,
                                  int32_t lost_remove_idx[5],
                                  uint8_t *new_growth_mask)
{
    int lost_remove = 0;
    uint8_t rm = 0;
    for (int i = 0; i < 5; ++i) {
        if (node_coord_is_zero(q, i)) continue;
        const Node pred = node_minus(q, i);
        const int pslot = nodeset_find_slot(&s->remove, &pred);
        if (pslot >= 0) {
            ++lost_remove;
            rm |= (uint8_t)(1u << i);
            lost_remove_idx[i] = s->remove.slot[pslot] - 1;
        } else {
            lost_remove_idx[i] = -1;
        }
    }
    *new_gm = s->remove.count + 1 - lost_remove;
    if (lost_remove_mask) *lost_remove_mask = rm;

    int new_growth = s->growth.count - 1;
    uint8_t gm = 0;
    for (int i = 0; i < 5; ++i) {
        /* q is necessarily the missing predecessor in direction i, so the
           other two predecessor checks are sufficient. */
        const Node candidate = node_plus(q, i);
        bool ok = true;
        for (int j = 0; j < 5; ++j) {
            if (j == i) continue;
            if (node_coord_is_zero(q, j)) continue;
            const Node pred = node_minus(&candidate, j);
            if (!part_contains(s, &pred)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            ++new_growth;
            gm |= (uint8_t)(1u << i);
        }
    }
    *new_gp = new_growth;
    if (new_growth_mask) *new_growth_mask = gm;
}

static inline void predict_remove(const ChainState *s, const Node *x,
                                   int *new_gp, int *new_gm,
                                   uint8_t *lost_growth_mask,
                                   int32_t lost_growth_idx[5],
                                   uint8_t *new_remove_mask)
{
    int lost_growth = 0;
    uint8_t gm = 0;
    for (int i = 0; i < 5; ++i) {
        const Node succ = node_plus(x, i);
        const int sslot = nodeset_find_slot(&s->growth, &succ);
        if (sslot >= 0) {
            ++lost_growth;
            gm |= (uint8_t)(1u << i);
            lost_growth_idx[i] = s->growth.slot[sslot] - 1;
        } else {
            lost_growth_idx[i] = -1;
        }
    }
    *new_gp = s->growth.count + 1 - lost_growth;
    if (lost_growth_mask) *lost_growth_mask = gm;

    int new_remove = s->remove.count - 1;
    uint8_t rm = 0;
    for (int i = 0; i < 5; ++i) {
        if (node_coord_is_zero(x, i)) continue;
        const Node pred = node_minus(x, i);
        bool ok = true;
        for (int j = 0; j < 5; ++j) {
            if (j == i) continue;
            const Node succ = node_plus(&pred, j);
            if (part_contains(s, &succ)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            ++new_remove;
            rm |= (uint8_t)(1u << i);
        }
    }
    *new_gm = new_remove;
    if (new_remove_mask) *new_remove_mask = rm;
}

static inline bool nodeset_remove_pending(NodeSet *s, int32_t pending[5], int count)
{
    for (int i = 0; i < count; ++i) {
        const int target_idx = pending[i];
        if (target_idx < 0 || target_idx >= s->count) return false;
        const int last = s->count - 1;
        const int pos = s->item_slot[target_idx];
        if (!nodeset_remove_slot(s, pos)) return false;
        if (target_idx != last) {
            /* nodeset_remove_slot() moved the former last item into target_idx.
               Repair only future pending indices. This preserves the exact
               dimension-order deletion sequence of the reference. */
            for (int j = i + 1; j < count; ++j)
                if (pending[j] == last) {
                    pending[j] = target_idx;
                    break;
                }
        }
    }
    return true;
}

static inline void apply_add(ChainState *s, const Node *q, uint64_t qhash, int growth_slot,
                             uint8_t lost_remove_mask, const int32_t lost_remove_idx[5],
                             uint8_t new_growth_mask)
{
    if (s->level >= N - 1) die("Attempted to grow beyond N");
    if (!nodeset_remove_slot(&s->growth, growth_slot)) die("Added node was not a growth point");
    nodeset_add_absent_h(&s->remove, q, qhash);

    int32_t pending[5]; int count = 0;
    for (int i = 0; i < 5; ++i)
        if (lost_remove_mask & (uint8_t)(1u << i)) pending[count++] = lost_remove_idx[i];
    if (!nodeset_remove_pending(&s->remove, pending, count))
        die("Predicted removable node missing");

    ++s->level;
    nodeset_add_absent_h(&s->part_set, q, qhash);

    for (int i = 0; i < 5; ++i)
        if (new_growth_mask & (uint8_t)(1u << i)) {
            const Node succ = node_plus(q, i);
            nodeset_add_absent(&s->growth, &succ);
        }
}

static inline void apply_remove(ChainState *s, const Node *x, uint64_t xhash, int remove_slot,
                                uint8_t lost_growth_mask, const int32_t lost_growth_idx[5],
                                uint8_t new_remove_mask)
{
    if (s->level <= 0) die("Attempted to remove last box");
    if (!nodeset_remove_slot(&s->remove, remove_slot)) die("Removed node was not removable");
    nodeset_add_absent_h(&s->growth, x, xhash);

    int32_t pending[5]; int count = 0;
    for (int i = 0; i < 5; ++i)
        if (lost_growth_mask & (uint8_t)(1u << i)) pending[count++] = lost_growth_idx[i];
    if (!nodeset_remove_pending(&s->growth, pending, count))
        die("Predicted growth node missing");

    if (!nodeset_remove_h(&s->part_set, x, xhash)) die("Failed removing partition node");

    for (int i = 0; i < 5; ++i)
        if (new_remove_mask & (uint8_t)(1u << i)) {
            const Node pred = node_minus(x, i);
            nodeset_add_absent(&s->remove, &pred);
        }
    --s->level;
}

static inline void apply_add_unpredicted(ChainState *s, const Node *q, int growth_slot)
{
    const int qidx = s->growth.slot[growth_slot] - 1;
    const uint64_t qhash = s->growth.item_hash[qidx];
    uint8_t rm = 0, gm = 0;
    int32_t ridx[5] = {-1, -1, -1, -1, -1};
    int new_gp, new_gm;
    predict_add(s, q, &new_gp, &new_gm, &rm, ridx, &gm);
    apply_add(s, q, qhash, growth_slot, rm, ridx, gm);
}

// Grow from the origin to a requested level. Used only to obtain an initial
// state for a restricted WL window. The random path is intentionally arbitrary;
// WL adaptation removes dependence on this initial condition before production.
static void build_state_at_level(ChainState *s, RNG *rng, int target_level)
{
    build_initial_state(s);
    while (s->level < target_level) {
        if (s->growth.count <= 0) die("No growth point while building target level");
        const int qidx = (int)rand_index(rng, (uint32_t)s->growth.count);
        Node q = s->growth.items[qidx];
        apply_add_unpredicted(s, &q, s->growth.item_slot[qidx]);
    }
}

// -----------------------------------------------------------------------------
// MH step restricted to one level window [lo, hi].
// log_weight is indexed by absolute level, with arbitrary additive constant.
// -----------------------------------------------------------------------------

static inline long double proposal_up(int level, int lo, int hi)
{
    if (level >= hi) return 0.0L;
    if (level == lo) return 1.0L;
    return 0.5L;
}

static inline long double proposal_down(int level, int lo, int hi)
{
    if (level <= lo) return 0.0L;
    if (level == hi) return 1.0L;
    return 0.5L;
}

static bool mh_step_window(ChainState *s, RNG *rng,
                           const long double *log_weight,
                           int lo, int hi)
{
    const int old_level = s->level;
    const long double pu = proposal_up(old_level, lo, hi);
    const long double pd = proposal_down(old_level, lo, hi);

    bool up;
    if (pd == 0.0L) up = true;
    else if (pu == 0.0L) up = false;
    else up = rand_bit_half(rng);

    if (up) {
        if (old_level >= hi || s->growth.count <= 0) die("Invalid upward proposal");
        const int new_level = old_level + 1;
        const int qidx = (int)rand_index(rng, (uint32_t)s->growth.count);
        const Node q = s->growth.items[qidx];
        const int qslot = s->growth.item_slot[qidx];
        const uint64_t qhash = s->growth.item_hash[qidx];

        int new_gp, new_gm;
        uint8_t lost_remove_mask, new_growth_mask;
        int32_t lost_remove_idx[5] = {-1, -1, -1, -1, -1};
        predict_add(s, &q, &new_gp, &new_gm,
                    &lost_remove_mask, lost_remove_idx, &new_growth_mask);
        if (new_gp <= 0 || new_gm <= 0) die("Invalid predicted degree after growth");

        const long double log_q_forward =
            (pu == 1.0L) ? g_log_inv_count[s->growth.count]
                         : g_log_half_over_count[s->growth.count];
        const long double reverse_p = proposal_down(new_level, lo, hi);
        const long double log_q_reverse =
            (reverse_p == 1.0L) ? g_log_inv_count[new_gm]
                                : g_log_half_over_count[new_gm];

        long double log_alpha =
            (log_weight[new_level - lo] - log_weight[old_level - lo]) +
            log_q_reverse - log_q_forward;
        if (log_alpha > 0.0L) log_alpha = 0.0L;

        long double u = rand_unit(rng);
        while (u == 0.0L) u = rand_unit(rng);
        if (log_alpha == 0.0L || logl(u) < log_alpha) {
            apply_add(s, &q, qhash, qslot,
                      lost_remove_mask, lost_remove_idx, new_growth_mask);
            return true;
        }
        return false;
    }

    if (old_level <= lo || s->remove.count <= 0) die("Invalid downward proposal");
    const int new_level = old_level - 1;
    const int xidx = (int)rand_index(rng, (uint32_t)s->remove.count);
    const Node x = s->remove.items[xidx];
    const int xslot = s->remove.item_slot[xidx];
    const uint64_t xhash = s->remove.item_hash[xidx];

    int new_gp, new_gm;
    uint8_t lost_growth_mask, new_remove_mask;
    int32_t lost_growth_idx[5] = {-1, -1, -1, -1, -1};
    predict_remove(s, &x, &new_gp, &new_gm,
                   &lost_growth_mask, lost_growth_idx, &new_remove_mask);
    if (new_gp <= 0 || new_gm <= 0) die("Invalid predicted degree after removal");

    const long double log_q_forward =
        (pd == 1.0L) ? g_log_inv_count[s->remove.count]
                     : g_log_half_over_count[s->remove.count];
    const long double reverse_p = proposal_up(new_level, lo, hi);
    const long double log_q_reverse =
        (reverse_p == 1.0L) ? g_log_inv_count[new_gp]
                            : g_log_half_over_count[new_gp];

    long double log_alpha =
        (log_weight[new_level - lo] - log_weight[old_level - lo]) +
        log_q_reverse - log_q_forward;
    if (log_alpha > 0.0L) log_alpha = 0.0L;

    long double u = rand_unit(rng);
    while (u == 0.0L) u = rand_unit(rng);
    if (log_alpha == 0.0L || logl(u) < log_alpha) {
        apply_remove(s, &x, xhash, xslot,
                         lost_growth_mask, lost_growth_idx, new_remove_mask);
        return true;
    }
    return false;
}


// -----------------------------------------------------------------------------
// 1/t Wang-Landau
// -----------------------------------------------------------------------------

static bool histogram_flat(const uint64_t *hist, int n, long double fraction)
{
    uint64_t min_h = UINT64_MAX;
    uint64_t max_h = 0;
    for (int i = 0; i < n; ++i) {
        if (hist[i] == 0) return false;
        if (hist[i] < min_h) min_h = hist[i];
        if (hist[i] > max_h) max_h = hist[i];
    }
    return (long double)min_h >= fraction * (long double)max_h;
}

static uint64_t base_seed_for_window(uint64_t root_seed, int window_id)
{
    return splitmix64(root_seed ^
                      (0x9e3779b97f4a7c15ULL * (uint64_t)(window_id + 1)));
}

static uint64_t run_window_wl(const Window *w, uint64_t seed,
                              long double *log_g, long double *final_lnf,
                              uint64_t one_over_t_steps)
{
    const int len = w->len;
    ChainState s;
    RNG rng;
    seed_rng(&rng, seed);
    build_state_at_level(&s, &rng, w->lo);

    for (int i = 0; i < len; ++i) log_g[i] = 0.0L;

    uint64_t *hist = (uint64_t *)calloc((size_t)len, sizeof(uint64_t));
    if (!hist) die("Out of memory for WL histogram");

    long double *log_weight = (long double *)malloc((size_t)len * sizeof(long double));
    if (!log_weight) die("Out of memory for WL weights");

    for (int i = 0; i < len; ++i) log_weight[i] = 0.0L;

    long double F = WL_INITIAL_LNF;
    uint64_t t = 0;
    bool one_over_t = false;
    unsigned flat_stages = 0;
    uint64_t check_interval = WL_CHECK_INTERVAL;
    const uint64_t small_window_check = 1000ULL * (uint64_t)(len > 0 ? len : 1);
    if (check_interval > small_window_check) check_interval = small_window_check;
    if (check_interval == 0) check_interval = 1;

    hist[0] = 1;
    log_g[0] += F;
    ++t;

    while (!one_over_t) {
        if (t >= WL_MAX_STEPS) die("WL_MAX_STEPS reached before 1/t transition");

        (void)mh_step_window(&s, &rng, log_weight, w->lo, w->hi);
        const int idx = s.level - w->lo;

        // WL update at the newly visited state.
        log_g[idx] += F;
        log_weight[idx] = -log_g[idx];
        ++hist[idx];
        ++t;

        /* Science-grade WL preconditioning: require repeated, stringent
           histogram-flat stages.  A stage is accepted only after every level
           has at least WL_MIN_VISITS_PER_BIN visits and the min/max ratio is
           at least WL_FLATNESS.  The histogram is then reset and the
           modification factor is halved.  We require WL_MIN_FLAT_STAGES such
           stages before allowing the Belardinelli-Pereyra 1/t regime. */
        if ((t % check_interval) == 0) {
            bool enough_visits = true;
            for (int i = 0; i < len; ++i) {
                if (hist[i] < WL_MIN_VISITS_PER_BIN) {
                    enough_visits = false;
                    break;
                }
            }
            if (enough_visits && histogram_flat(hist, len, WL_FLATNESS)) {
                ++flat_stages;
                F *= 0.5L;
                memset(hist, 0, (size_t)len * sizeof(uint64_t));
            }
        }

        if (flat_stages >= WL_MIN_FLAT_STAGES &&
            t >= (uint64_t)len &&
            F <= WL_1T_SCALE * (long double)len / (long double)t) {
            one_over_t = true;
            break;
        }
    }

    // Genuine Belardinelli-Pereyra 1/t regime. The scale is normally 1.
    for (uint64_t j = 0; j < one_over_t_steps; ++j) {
        if (t >= WL_MAX_STEPS) die("WL_MAX_STEPS reached during 1/t phase");
        F = WL_1T_SCALE * (long double)len / (long double)t;

        (void)mh_step_window(&s, &rng, log_weight, w->lo, w->hi);
        const int idx = s.level - w->lo;
        log_g[idx] += F;
        log_weight[idx] = -log_g[idx];
        ++hist[idx];
        ++t;

#if CHECK_STATE
        if ((t % CHECK_INTERVAL) == 0) consistency_check(&s);
#endif
    }

    // Remove the irrelevant additive constant.
    const long double offset = log_g[0];
    for (int i = 0; i < len; ++i) log_g[i] -= offset;

    if (final_lnf) *final_lnf =
        WL_1T_SCALE * (long double)len / (long double)t;

    free(log_weight);
    free(hist);
    free_state(&s);
    return t;
}

static void run_window_production(const Window *w, const long double *log_g,
                                  uint64_t seed, WindowResult *out,
                                  uint64_t burn_in_steps,
                                  uint64_t production_steps)
{
    const int len = w->len;
    memset(out, 0, sizeof *out);
    out->hits = (uint64_t *)calloc((size_t)len, sizeof(uint64_t));
    out->gp_sum = (long double *)calloc((size_t)len, sizeof(long double));
    out->gm_sum = (long double *)calloc((size_t)len, sizeof(long double));
    if (!out->hits || !out->gp_sum || !out->gm_sum) die("Out of memory for production arrays");

    RNG rng;
    seed_rng(&rng, seed);
    ChainState s;
    build_state_at_level(&s, &rng, w->lo);

    long double *log_weight = (long double *)malloc((size_t)len * sizeof(long double));
    if (!log_weight) die("Out of memory for production weights");
    for (int i = 0; i < len; ++i) log_weight[i] = -log_g[i];

    uint64_t accepted = 0;
    uint64_t attempted = 0;

    for (uint64_t i = 0; i < burn_in_steps; ++i) {
        if (mh_step_window(&s, &rng, log_weight, w->lo, w->hi)) ++accepted;
        ++attempted;
    }

    for (uint64_t i = 0; i < production_steps; ++i) {
        const int idx = s.level - w->lo;
        ++out->hits[idx];
        out->gp_sum[idx] += (long double)s.growth.count;
        out->gm_sum[idx] += (long double)s.remove.count;

        if (mh_step_window(&s, &rng, log_weight, w->lo, w->hi)) ++accepted;
        ++attempted;

#if CHECK_STATE
        if ((i + 1) % CHECK_INTERVAL == 0) consistency_check(&s);
#endif
    }

    out->production_steps = production_steps;
    out->accepted = accepted;
    out->attempted = attempted;
    free(log_weight);
    free_state(&s);
}

static void free_window_result(WindowResult *r)
{
    free(r->hits);
    free(r->gp_sum);
    free(r->gm_sum);
    memset(r, 0, sizeof *r);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void)
{
    struct timespec t0, t1;
    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0) die("clock_gettime failed");

    omp_set_dynamic(0);
    omp_set_num_threads(MC_THREADS);

    if (NUM_RUNS < 1) die("NUM_RUNS must be >= 1");

    /* One-time packing-width selection. Five coordinates are packed into
       128 bits; for 15000--20000, this uses 15 bits per coordinate. */
    {
        unsigned bits = 1;
        while (((__uint128_t)1 << bits) <= (uint64_t)N) ++bits;
        if (bits > 25) die("N is too large for 128-bit 5-coordinate packing");
        g_coord_bits = bits;
        g_coord_mask = ((__uint128_t)1 << bits) - 1;
        for (int d = 0; d < 5; ++d)
            g_coord_step[d] = (__uint128_t)1 << (d * bits);
    }

    g_log_count_cap = (size_t)(5 * N + 8);
    g_log_inv_count = (long double *)malloc((g_log_count_cap + 1) * sizeof(long double));
    g_log_half_over_count = (long double *)malloc((g_log_count_cap + 1) * sizeof(long double));
    if (!g_log_inv_count || !g_log_half_over_count) die("Out of memory for proposal-log tables");
    for (size_t k = 1; k <= g_log_count_cap; ++k) {
        const long double kk = (long double)k;
        g_log_inv_count[k] = logl(1.0L / kk);
        g_log_half_over_count[k] = logl(0.5L / kk);
    }

    const bool runtime_seed_used = (RNG_SEED == 0ULL);
    uint64_t root_seed = RNG_SEED;
    if (runtime_seed_used) root_seed = runtime_seed();

    if (runtime_seed_used) {
        FILE *seed_fp = fopen(SEED_FILE, "w");
        if (!seed_fp) die("Unable to create seed.txt");
        fprintf(seed_fp, "root_rng_seed_hex=0x%016" PRIx64 "\n", root_seed);
        fprintf(seed_fp, "root_rng_seed_decimal=%" PRIu64 "\n", root_seed);
        fprintf(seed_fp, "RNG_SEED was 0ULL; all subsequent run/window seeds are deterministic descendants of this root seed.\n");
        if (fclose(seed_fp) != 0) die("Unable to close seed.txt");
    }

    fprintf(stderr, "Root RNG seed = 0x%016" PRIx64 " (%s)\n",
            root_seed, runtime_seed_used ? "runtime" : "fixed");
    fprintf(stderr, "OpenMP threads = %d\n", MC_THREADS);
    fprintf(stderr, "Independent Monte Carlo runs = %d\n", NUM_RUNS);

    const int first_level_raw = (WL_REQUIRED_LEVEL_SPAN > WL_FIRST_WINDOW_PAD_THRESHOLD)
        ? (KNOWN_N - 1 - WL_WINDOW_OVERLAP)
        : (KNOWN_N - 1);
    const int first_level = (first_level_raw < 0) ? 0 : first_level_raw;
    const int last_level = N - 1;
    const int step = WL_WINDOW_WIDTH - WL_WINDOW_OVERLAP;

    int max_windows = (last_level - first_level) / step + 2;
    Window *windows = (Window *)malloc((size_t)max_windows * sizeof(Window));
    if (!windows) die("Out of memory for windows");

    int nw = 0;
    int start = first_level;
    if (start < 0) die("Internal error: negative window start");
    while (start <= last_level) {
        int hi = start + WL_WINDOW_WIDTH - 1;
        if (hi > last_level) hi = last_level;
        windows[nw].lo = start;
        windows[nw].hi = hi;
        windows[nw].len = hi - start + 1;
        ++nw;
        if (hi == last_level) break;
        start += step;
    }

    fprintf(stderr,
            "N=%d, anchor=%d, required levels=%d..%d, windows/run=%d, width=%d, overlap=%d\n",
            N, KNOWN_N, first_level, last_level, nw,
            WL_WINDOW_WIDTH, WL_WINDOW_OVERLAP);

    const int nlevels = last_level - first_level + 1;
    uint64_t *pooled_hits = (uint64_t *)calloc((size_t)nlevels, sizeof(uint64_t));
    long double *pooled_gp_sum = (long double *)calloc((size_t)nlevels, sizeof(long double));
    long double *pooled_gm_sum = (long double *)calloc((size_t)nlevels, sizeof(long double));
    if (!pooled_hits || !pooled_gp_sum || !pooled_gm_sum)
        die("Out of memory for pooled statistics");

    uint64_t grand_wl_steps = 0;
    uint64_t grand_prod_steps = 0;
    uint64_t grand_accepted = 0;
    uint64_t grand_attempted = 0;

    // Direct independent-run statistics for Delta_4(n) itself.
    // Array index i corresponds to Delta_4(KNOWN_N + 1 + i).
    const size_t nDelta = (size_t)(N - KNOWN_N);
    long double *run_Delta_mean = (long double *)calloc(nDelta, sizeof(long double));
    long double *run_Delta_M2 = (long double *)calloc(nDelta, sizeof(long double));
    uint64_t *run_Delta_count = (uint64_t *)calloc(nDelta, sizeof(uint64_t));
    if (!run_Delta_mean || !run_Delta_M2 || !run_Delta_count)
        die("Out of memory for Delta_4 error statistics");

    const uint64_t min_wl_steps = 250000ULL;
    const uint64_t min_prod_steps = 250000ULL;
    const uint64_t min_burn_steps = 10000ULL;

    for (int run = 0; run < NUM_RUNS; ++run) {
        const uint64_t run_seed =
            splitmix64(root_seed ^
                       (0x9e3779b97f4a7c15ULL * (uint64_t)(run + 1)));

        fprintf(stderr, "\n=== Independent run %d/%d: seed = 0x%016" PRIx64 " ===\n",
                run + 1, NUM_RUNS, run_seed);

        WindowResult *results =
            (WindowResult *)calloc((size_t)nw, sizeof(WindowResult));
        long double **local_log_g =
            (long double **)calloc((size_t)nw, sizeof(long double *));
        if (!results || !local_log_g) die("Out of memory for run window results");

#pragma omp parallel for schedule(dynamic,1)
        for (int wi = 0; wi < nw; ++wi) {
            uint64_t s_wl = base_seed_for_window(run_seed, 2 * wi);
            uint64_t s_prod = base_seed_for_window(run_seed, 2 * wi + 1);

            local_log_g[wi] =
                (long double *)malloc((size_t)windows[wi].len * sizeof(long double));
            if (!local_log_g[wi]) die("Out of memory for local log_g");

            const uint64_t wl_steps_for_window =
                (uint64_t)fmax((long double)min_wl_steps,
                               ((long double)WL_1T_STEPS *
                                (long double)windows[wi].len) /
                               (long double)WL_STEP_REFERENCE_WIDTH);
            const uint64_t prod_steps_for_window =
                (uint64_t)fmax((long double)min_prod_steps,
                               ((long double)PROD_STEPS *
                                (long double)windows[wi].len) /
                               (long double)WL_STEP_REFERENCE_WIDTH);
            const uint64_t burn_steps_for_window =
                (uint64_t)fmax((long double)min_burn_steps,
                               ((long double)PROD_BURN_IN *
                                (long double)windows[wi].len) /
                               (long double)WL_STEP_REFERENCE_WIDTH);

            long double wl_final_f = 0.0L;
            uint64_t wl_steps = run_window_wl(&windows[wi], s_wl,
                                              local_log_g[wi], &wl_final_f,
                                              wl_steps_for_window);
            run_window_production(&windows[wi], local_log_g[wi], s_prod,
                                  &results[wi], burn_steps_for_window,
                                  prod_steps_for_window);
            results[wi].wl_steps = wl_steps;
            results[wi].final_lnf = wl_final_f;

#if VERBOSE_PROGRESS
#pragma omp critical
            {
                fprintf(stderr,
                        "run %d window %d/%d [%d,%d]: WL=%" PRIu64
                        ", final F=%.3Le, prod=%" PRIu64
                        ", acceptance=%.5Lf\n",
                        run + 1, wi + 1, nw,
                        windows[wi].lo, windows[wi].hi,
                        wl_steps, results[wi].final_lnf,
                        results[wi].production_steps,
                        results[wi].attempted ?
                        (long double)results[wi].accepted /
                        (long double)results[wi].attempted : 0.0L);
            }
#endif
        }

        uint64_t *run_hits =
            (uint64_t *)calloc((size_t)nlevels, sizeof(uint64_t));
        long double *run_gp_sum =
            (long double *)calloc((size_t)nlevels, sizeof(long double));
        long double *run_gm_sum =
            (long double *)calloc((size_t)nlevels, sizeof(long double));
        if (!run_hits || !run_gp_sum || !run_gm_sum)
            die("Out of memory for per-run statistics");

        uint64_t run_wl_steps = 0;
        uint64_t run_prod_steps = 0;
        uint64_t run_accepted = 0;
        uint64_t run_attempted = 0;

        for (int wi = 0; wi < nw; ++wi) {
            const Window *w = &windows[wi];
            const WindowResult *r = &results[wi];
            run_wl_steps += r->wl_steps;
            run_prod_steps += r->production_steps;
            run_accepted += r->accepted;
            run_attempted += r->attempted;

            for (int j = 0; j < w->len; ++j) {
                const int global_idx = (w->lo - first_level) + j;
                run_hits[global_idx] += r->hits[j];
                run_gp_sum[global_idx] += r->gp_sum[j];
                run_gm_sum[global_idx] += r->gm_sum[j];

                pooled_hits[global_idx] += r->hits[j];
                pooled_gp_sum[global_idx] += r->gp_sum[j];
                pooled_gm_sum[global_idx] += r->gm_sum[j];
            }
        }

        grand_wl_steps += run_wl_steps;
        grand_prod_steps += run_prod_steps;
        grand_accepted += run_accepted;
        grand_attempted += run_attempted;

        // Direct run-level Delta_4(n) statistics.  This is exactly the local
        // quantity used in the final scientific fit and avoids reconstructing
        // an entire p_4(n) curve just to difference logs afterward.
#if NUM_RUNS > 1
        for (int n = KNOWN_N; n < N; ++n) {
            const int idx_n = n - 1 - first_level;
            const int idx_np1 = n - first_level;
            if (idx_n < 0 || idx_np1 < 0 || idx_n >= nlevels || idx_np1 >= nlevels ||
                run_hits[idx_n] == 0 || run_hits[idx_np1] == 0) {
                continue;
            }
            const long double gp_n =
                run_gp_sum[idx_n] / (long double)run_hits[idx_n];
            const long double gm_np1 =
                run_gm_sum[idx_np1] / (long double)run_hits[idx_np1];
            if (!(gp_n > 0.0L) || !(gm_np1 > 0.0L) ||
                !isfinite(gp_n) || !isfinite(gm_np1)) {
                continue;
            }
            const long double sample_delta = logl(gp_n) - logl(gm_np1);
            const size_t ai = (size_t)(n - KNOWN_N);
            const long double k = (long double)(run_Delta_count[ai] + 1U);
            const long double delta = sample_delta - run_Delta_mean[ai];
            run_Delta_mean[ai] += delta / k;
            run_Delta_M2[ai] += delta * (sample_delta - run_Delta_mean[ai]);
            run_Delta_count[ai] += 1U;
        }
#endif

        for (int wi = 0; wi < nw; ++wi) {
            free_window_result(&results[wi]);
            free(local_log_g[wi]);
        }
        free(local_log_g);
        free(results);
        free(run_hits);
        free(run_gp_sum);
        free(run_gm_sum);
    }

    printf("N = %d\n", N);
    printf("KNOWN_N = %d, KNOWN_P4 = %.0Lf\n", KNOWN_N, KNOWN_P4);
    printf("NUM_RUNS = %d\n", NUM_RUNS);
    printf("MC_THREADS = %d\n", MC_THREADS);
    printf("Number of windows per run = %d, window width = %d, overlap = %d\n",
           nw, WL_WINDOW_WIDTH, WL_WINDOW_OVERLAP);
    printf("Reference work per 128 levels: WL_1T=%llu, PROD=%llu, BURN=%llu\n",
           (unsigned long long)WL_1T_STEPS,
           (unsigned long long)PROD_STEPS,
           (unsigned long long)PROD_BURN_IN);
    printf("Per-window work scales linearly with actual window length.\n");
    printf("Total WL steps over all runs = %" PRIu64 "\n", grand_wl_steps);
    printf("Total production steps over all runs = %" PRIu64 "\n", grand_prod_steps);
    printf("Pooled production acceptance = %.6Lf\n",
           grand_attempted ?
               (long double)grand_accepted / (long double)grand_attempted : 0.0L);

    if (!(KNOWN_P4 > 0.0L) || !isfinite(KNOWN_P4)) {
        fprintf(stderr, "KNOWN_P4 must be finite and positive\n");
        return EXIT_FAILURE;
    }

    FILE *csv_fp = fopen(DELTA4_CSV_FILE, "w");
    if (!csv_fp) die("Unable to create MC4D_Delta4.csv");
    if (fprintf(csv_fp, "n,Delta_4,stat_err_percentage\n") < 0) {
        fclose(csv_fp);
        die("Unable to write MC4D_Delta4.csv header");
    }

    // Write the direct Delta_4(n) observable and its independent-run statistical
    // error as a percentage.  No Monte Carlo data columns are printed to stdout.
    long double log_p3_reconstructed = logl(KNOWN_P4);
    for (int n = KNOWN_N; n < N; ++n) {
        const int idx_n = n - 1 - first_level;
        const int idx_np1 = n - first_level;
        if (idx_n < 0 || idx_np1 < 0 ||
            idx_n >= nlevels || idx_np1 >= nlevels ||
            pooled_hits[idx_n] == 0 || pooled_hits[idx_np1] == 0) {
            fclose(csv_fp);
            fprintf(stderr, "Missing pooled statistics for Delta_4(%d)\n", n + 1);
            return EXIT_FAILURE;
        }

        const long double gp_n =
            pooled_gp_sum[idx_n] / (long double)pooled_hits[idx_n];
        const long double gm_np1 =
            pooled_gm_sum[idx_np1] / (long double)pooled_hits[idx_np1];
        if (!(gp_n > 0.0L) || !(gm_np1 > 0.0L) ||
            !isfinite(gp_n) || !isfinite(gm_np1)) {
            fclose(csv_fp);
            fprintf(stderr, "Invalid pooled broad-histogram averages for Delta_4(%d)\n", n + 1);
            return EXIT_FAILURE;
        }

        const long double delta3 = logl(gp_n) - logl(gm_np1);
        long double sigma_delta = 0.0L;
#if NUM_RUNS > 1
        const size_t ai = (size_t)(n - KNOWN_N);
        if (run_Delta_count[ai] >= 2) {
            const long double c = (long double)run_Delta_count[ai];
            const long double var_delta = run_Delta_M2[ai] / (c - 1.0L);
            sigma_delta = sqrtl(var_delta / c);
        }
#endif
        const long double stat_err_pct =
            (delta3 != 0.0L) ? 100.0L * sigma_delta / fabsl(delta3) : 0.0L;

        if (fprintf(csv_fp, "%d,%.18Le,%.12Lf\n",
                    n + 1, delta3, stat_err_pct) < 0) {
            fclose(csv_fp);
            die("Unable to write MC4D_Delta4.csv data");
        }

        log_p3_reconstructed += delta3;
    }

    if (fclose(csv_fp) != 0) die("Unable to close MC4D_Delta4.csv");

    // Reconstruct only p_4(N), once, for the console validation/result line.
    printf("Reconstructed p_4(%d) = ", N);
    if (log_p3_reconstructed <= logl(LDBL_MAX)) {
        printf("%.18Le\n", expl(log_p3_reconstructed));
    } else {
        printf("10^(%.18Lf) [log-space]\n",
               log_p3_reconstructed / logl(10.0L));
    }

    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) die("clock_gettime failed at end");
    const long double elapsed =
        (long double)(t1.tv_sec - t0.tv_sec) +
        1.0e-9L * (long double)(t1.tv_nsec - t0.tv_nsec);
    printf("Total computation time is: %.6Lf seconds\n", elapsed);

    free(windows);
    free(pooled_hits);
    free(pooled_gp_sum);
    free(pooled_gm_sum);
    free(run_Delta_mean);
    free(run_Delta_M2);
    free(run_Delta_count);
    free(g_log_inv_count);
    free(g_log_half_over_count);
    return EXIT_SUCCESS;
}
