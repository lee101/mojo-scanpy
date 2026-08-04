"""Hot kernels for mojo-scanpy over contiguous float64 buffers."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def distance2(a: Ptr, b: Ptr, d: Int) -> Float64:
    var total = 0.0
    comptime W = simd_width_of[DType.float64]()
    var j = 0
    var vector_end = d - d % W
    for j in range(0, vector_end, W):
        var delta = a.load[width=W](j) - b.load[width=W](j)
        total += (delta * delta).reduce_add()
    for j in range(vector_end, d):
        var delta = a[j] - b[j]
        total += delta * delta
    return total


def knn(
    train: Ptr, query: Ptr, indices: Ptr, distances: Ptr,
    n: Int, d: Int, m: Int, k: Int, exclude_self: Int,
):
    for row in range(m):
        var base = row * k
        for slot in range(k):
            distances[base + slot] = 1.7976931348623157e308
            indices[base + slot] = -1.0
        for candidate in range(n):
            if exclude_self != 0 and candidate == row:
                continue
            var value = distance2(query + row * d, train + candidate * d, d)
            if value >= distances[base + k - 1]:
                continue
            var slot = k - 1
            while slot > 0 and distances[base + slot - 1] > value:
                distances[base + slot] = distances[base + slot - 1]
                indices[base + slot] = indices[base + slot - 1]
                slot -= 1
            distances[base + slot] = value
            indices[base + slot] = Float64(candidate)


def covariance(x: Ptr, mean: Ptr, matrix: Ptr, n: Int, d: Int):
    for j in range(d):
        mean[j] = 0.0
    for row in range(n):
        for j in range(d):
            mean[j] += x[row * d + j]
    for j in range(d):
        mean[j] /= Float64(n)
    for i in range(d * d):
        matrix[i] = 0.0
    for row in range(n):
        for i in range(d):
            var left = x[row * d + i] - mean[i]
            for j in range(i + 1):
                matrix[i * d + j] += left * (x[row * d + j] - mean[j])
    var denom = Float64(n - 1) if n > 1 else 1.0
    for i in range(d):
        for j in range(i + 1):
            var value = matrix[i * d + j] / denom
            matrix[i * d + j] = value
            matrix[j * d + i] = value


def jacobi(matrix: Ptr, vectors: Ptr, d: Int):
    for i in range(d):
        for j in range(d):
            vectors[i * d + j] = 1.0 if i == j else 0.0
    for _ in range(80):
        var off = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                off += matrix[i * d + j] * matrix[i * d + j]
        if off < 1e-26:
            return
        for left in range(d):
            for right in range(left + 1, d):
                var mixed = matrix[left * d + right]
                if mixed == 0.0:
                    continue
                var theta = (matrix[right * d + right] - matrix[left * d + left]) / (2.0 * mixed)
                var sign = 1.0 if theta >= 0.0 else -1.0
                var tangent = sign / (abs(theta) + sqrt(theta * theta + 1.0))
                var cosine = 1.0 / sqrt(tangent * tangent + 1.0)
                var sine = tangent * cosine
                for r in range(d):
                    var a = matrix[r * d + left]
                    var b = matrix[r * d + right]
                    matrix[r * d + left] = cosine * a - sine * b
                    matrix[r * d + right] = sine * a + cosine * b
                for c in range(d):
                    var a = matrix[left * d + c]
                    var b = matrix[right * d + c]
                    matrix[left * d + c] = cosine * a - sine * b
                    matrix[right * d + c] = sine * a + cosine * b
                for r in range(d):
                    var a = vectors[r * d + left]
                    var b = vectors[r * d + right]
                    vectors[r * d + left] = cosine * a - sine * b
                    vectors[r * d + right] = sine * a + cosine * b


@export("msp_knn")
def msp_knn(
    train: Int, query: Int, indices: Int, distances: Int,
    n: Int, d: Int, m: Int, k: Int, exclude_self: Int,
) abi("C") -> Int:
    if train == 0 or query == 0 or indices == 0 or distances == 0:
        return -1
    if n < 1 or d < 1 or m < 1 or k < 1 or k > n:
        return -1
    if exclude_self != 0 and k >= n:
        return -1
    knn(p(train), p(query), p(indices), p(distances), n, d, m, k, exclude_self)
    return 0


@export("msp_pca_fit")
def msp_pca_fit(
    x: Int, mean: Int, components: Int, variance: Int, matrix: Int, vectors: Int,
    n: Int, d: Int, k: Int,
) abi("C") -> Float64:
    if x == 0 or mean == 0 or components == 0 or variance == 0 or matrix == 0 or vectors == 0:
        return -1.0
    if n < 2 or d < 1 or k < 1 or k > d:
        return -1.0
    var covariance_matrix = p(matrix)
    var eigenvectors = p(vectors)
    var output_variance = p(variance)
    var output_components = p(components)
    covariance(p(x), p(mean), covariance_matrix, n, d)
    jacobi(covariance_matrix, eigenvectors, d)
    var total = 0.0
    for i in range(d):
        total += covariance_matrix[i * d + i]
    for slot in range(k):
        var best = slot
        for candidate in range(slot + 1, d):
            if covariance_matrix[candidate * d + candidate] > covariance_matrix[best * d + best]:
                best = candidate
        if best != slot:
            var value = covariance_matrix[slot * d + slot]
            covariance_matrix[slot * d + slot] = covariance_matrix[best * d + best]
            covariance_matrix[best * d + best] = value
            for row in range(d):
                value = eigenvectors[row * d + slot]
                eigenvectors[row * d + slot] = eigenvectors[row * d + best]
                eigenvectors[row * d + best] = value
        output_variance[slot] = covariance_matrix[slot * d + slot]
        var pivot = 0
        for row in range(1, d):
            if abs(eigenvectors[row * d + slot]) > abs(eigenvectors[pivot * d + slot]):
                pivot = row
        var flip = -1.0 if eigenvectors[pivot * d + slot] < 0.0 else 1.0
        for row in range(d):
            output_components[slot * d + row] = flip * eigenvectors[row * d + slot]
    return total


@export("msp_pca_transform")
def msp_pca_transform(
    x: Int, mean: Int, components: Int, result: Int, n: Int, d: Int, k: Int,
) abi("C") -> Int:
    if x == 0 or mean == 0 or components == 0 or result == 0:
        return -1
    if n < 1 or d < 1 or k < 1 or k > d:
        return -1
    var input = p(x)
    var center = p(mean)
    var pcs = p(components)
    var destination = p(result)
    for row in range(n):
        for component in range(k):
            var value = 0.0
            for j in range(d):
                value += (input[row * d + j] - center[j]) * pcs[component * d + j]
            destination[row * k + component] = value
    return 0
