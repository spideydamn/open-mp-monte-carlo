#include "hit.h"

bool hit_test(float x, float y, float z) {
    return x * x * x * (x - 2) + 4 * (y * y + z * z) <= 0;
}

const float* get_axis_range() {
    return new float[]{0, 2, -0.649519052838329, 0.649519052838329, -0.649519052838329, 0.649519052838329};
}