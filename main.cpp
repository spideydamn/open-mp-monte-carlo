#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <omp.h>

#include "hit.h"

const uint64_t STARTS_NUMBER = 1;

bool is_number(const std::string& s);

void Parse(int argc, char** argv, uint64_t& threads_number, std::string& input_filename, std::string& output_filename) {
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--no-omp", 8)) {
            threads_number = 0;
        } else if (!strncmp(argv[i], "--omp-threads", 13)) {
            if (!strncmp(argv[i + 1], "default", 7)) {
                threads_number = omp_get_max_threads();
            } else if (is_number(argv[i + 1])) {
                threads_number = std::atoll(argv[i + 1]);
            } else {
                std::cerr << "parse error\n";
                exit(1);
            }
            ++i;
        } else if (!strncmp(argv[i], "--input", 7)) {
            input_filename = argv[i + 1];
            ++i;
        } else if (!strncmp(argv[i], "--output", 8)) {
            output_filename = argv[i + 1];
            ++i;
        } else {
            std::cerr << "parse error\n";
            exit(1);
        }
    }
}

bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();

    while (it != s.end() && std::isdigit(*it)) {
        ++it;
    }

    return !s.empty() && it == s.end();
}



double MonteCarlo(const char* input_filename, const char* output_filename) {
    double start = omp_get_wtime();

    const float* axis_range = get_axis_range();

    std::ifstream input(input_filename);
    uint64_t accuracy_number;
    input >> accuracy_number;
    input.close();

    uint64_t hits_sum = 0;
    uint32_t seed = 2281337;
    uint32_t hi, lo;
    float x, y, z;

    for (uint64_t i = 0; i < accuracy_number; ++i) {

        // ParkMillerRandom for x
        lo = 16807 * (seed & 0xFFFF);
        hi = 16807 * (seed >> 16);

        lo += (hi & 0x7FFF) << 16;
        lo += hi >> 15;

        if (lo > 0x7FFFFFFF) {
            lo -= 0x7FFFFFFF;
        }

        seed = lo;
        x = (float)lo / (float) 0x7FFFFFFF;

        // ParkMillerRandom for y
        lo = 16807 * (seed & 0xFFFF);
        hi = 16807 * (seed >> 16);

        lo += (hi & 0x7FFF) << 16;
        lo += hi >> 15;

        if (lo > 0x7FFFFFFF) {
            lo -= 0x7FFFFFFF;
        }

        seed = lo;
        y = (float)lo / (float) 0x7FFFFFFF;

        // ParkMillerRandom for z
        lo = 16807 * (seed & 0xFFFF);
        hi = 16807 * (seed >> 16);

        lo += (hi & 0x7FFF) << 16;
        lo += hi >> 15;

        if (lo > 0x7FFFFFFF) {
            lo -= 0x7FFFFFFF;
        }

        seed = lo;
        z = (float)lo / (float) 0x7FFFFFFF;

        hits_sum += hit_test(axis_range[0] + x * (axis_range[1] - axis_range[0]),
                         axis_range[2] + y * (axis_range[3] - axis_range[2]),
                         axis_range[4] + z * (axis_range[5] - axis_range[4]));
    }

    FILE* output = fopen(output_filename, "w+");
    fprintf(output, "%g\n",
            (axis_range[1] - axis_range[0]) * (axis_range[3] - axis_range[2]) * (axis_range[5] - axis_range[4]) *
            (float) hits_sum / (float) accuracy_number);
    fclose(output);

    return omp_get_wtime() - start;
}

double OMPMonteCarlo(uint64_t threads_number, const char* input_filename, const char* output_filename) {
    double start = omp_get_wtime();

    const float* axis_range = get_axis_range();

    std::ifstream input(input_filename);
    uint64_t accuracy_number;
    input >> accuracy_number;
    input.close();

    uint64_t hits_sum = 0;
    uint64_t hits = 0;
    uint64_t thread_number;
    uint32_t thread_seed;
    uint32_t hi, lo;
    float x, y, z;

    #pragma omp parallel firstprivate(hits) private(hi, lo, thread_number, thread_seed, x, y, z) default(shared) num_threads(threads_number)
    {
        thread_number = omp_get_thread_num();
        thread_seed = (thread_number + 1) * static_cast<uint64_t>(omp_get_wtime());

        #pragma omp for nowait schedule(static)
        for (uint64_t i = 0; i < accuracy_number; ++i) {

            // ParkMillerRandom for x
            lo = 16807 * (thread_seed & 0xFFFF);
            hi = 16807 * (thread_seed >> 16);

            lo += (hi & 0x7FFF) << 16;
            lo += hi >> 15;

            if (lo > 0x7FFFFFFF) {
                lo -= 0x7FFFFFFF;
            }

            thread_seed = lo;
            x = (float)lo / (float) 0x7FFFFFFF;

            // ParkMillerRandom for y
            lo = 16807 * (thread_seed & 0xFFFF);
            hi = 16807 * (thread_seed >> 16);

            lo += (hi & 0x7FFF) << 16;
            lo += hi >> 15;

            if (lo > 0x7FFFFFFF) {
                lo -= 0x7FFFFFFF;
            }

            thread_seed = lo;
            y = (float)lo / (float) 0x7FFFFFFF;

            // ParkMillerRandom for z
            lo = 16807 * (thread_seed & 0xFFFF);
            hi = 16807 * (thread_seed >> 16);

            lo += (hi & 0x7FFF) << 16;
            lo += hi >> 15;

            if (lo > 0x7FFFFFFF) {
                lo -= 0x7FFFFFFF;
            }

            thread_seed = lo;
            z = (float)lo / (float) 0x7FFFFFFF;

            hits += hit_test(axis_range[0] + x * (axis_range[1] - axis_range[0]),
                             axis_range[2] + y * (axis_range[3] - axis_range[2]),
                             axis_range[4] + z * (axis_range[5] - axis_range[4]));
        }

        #pragma omp atomic
        hits_sum += hits;
    }

    FILE* output = fopen(output_filename, "w+");
    fprintf(output, "%g\n",
            (axis_range[1] - axis_range[0]) * (axis_range[3] - axis_range[2]) * (axis_range[5] - axis_range[4]) *
            (float) hits_sum / (float) accuracy_number);
    fclose(output);

    return omp_get_wtime() - start;
}

void Execute(uint64_t threads_number, const std::string& input_filename, const std::string& output_filename) {
    double time = 0;

    for (uint64_t i = 0; i < STARTS_NUMBER; ++i) {
        if (threads_number) {
            time += OMPMonteCarlo(threads_number, input_filename.c_str(), output_filename.c_str());
        } else {
            time += MonteCarlo(input_filename.c_str(), output_filename.c_str());
        }
    }

    printf("Time (%i thread(s)): %g ms\n", (int32_t) threads_number, 1000 * time / STARTS_NUMBER);
}


int main(int argc, char** argv) {
    uint64_t threads_number;
    std::string input_filename;
    std::string output_filename;

    Parse(argc, argv, threads_number, input_filename, output_filename);
    Execute(threads_number, input_filename, output_filename);

    return 0;
}