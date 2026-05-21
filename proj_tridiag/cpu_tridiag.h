#include <stdexcept>

/**
 * @brief Lifted from Numerical Recipes.
 */
template <typename T>
void tridag(const T* a, const T* b, const T* c,
            const T* r, T* u, T* gam, int n)
{
    static_assert(std::is_floating_point<T>::value,
                  "tridag requires a floating-point type");
 
    if (b[0] == T(0))
        throw std::runtime_error("tridag: zero pivot at row 0");
 
    T bet = b[0];
    u[0] = r[0] / bet;
 
    for (int j = 1; j < n; ++j)
    {
        gam[j] = c[j-1] / bet;
        bet = b[j] - a[j] * gam[j];
        if (bet == T(0))
            throw std::runtime_error("tridag: zero pivot at row " + std::to_string(j));
        u[j] = (r[j] - a[j] * u[j-1]) / bet;
    }
 
    for (int j = n - 2; j >= 0; --j)
        u[j] -= gam[j+1] * u[j+1];
}

/**
 * @brief Cyclic tridiagonal solver, lifted from Numerical Recipes.
 * Solves the cyclic system with corner entries alpha (bottom-left)
 * and beta (top-right) via Sherman-Morrison decomposition.
 *
 * @param a, b, c  Tridiagonal diagonals, length n
 * @param alpha    Bottom-left corner element
 * @param beta     Top-right corner element
 * @param r        RHS vector, length n
 * @param x        Output vector, length n
 * @param bb       Scratch array, length n (modified diagonal)
 * @param u        Scratch array, length n (Sherman-Morrison RHS)
 * @param z        Scratch array, length n (Sherman-Morrison solution)
 * @param gam      Scratch array, length n (used by tridag)
 * @param n        System size
 */
template <typename T>
void cyclic(const T* a, const T* b, const T* c,
            T alpha, T beta,
            const T* r, T* x,
            T* bb, T* u, T* z, T* gam, int n)
{
    static_assert(std::is_floating_point<T>::value,
                  "cyclic requires a floating-point type");

    if (n <= 2)
        throw std::runtime_error("cyclic: n too small");

    T gamma = -b[0];
    bb[0] = b[0] - gamma;
    bb[n - 1] = b[n - 1] - alpha * beta / gamma;
    for (int i = 1; i < n - 1; ++i)
        bb[i] = b[i];

    tridag(a, bb, c, r, x, gam, n);

    u[0] = gamma;
    u[n - 1] = alpha;
    for (int i = 1; i < n - 1; ++i)
        u[i] = T(0);

    tridag(a, bb, c, u, z, gam, n);

    T fact = (x[0] + beta * x[n - 1] / gamma) /
             (T(1) + z[0] + beta * z[n - 1] / gamma);

    for (int i = 0; i < n; ++i)
        x[i] -= fact * z[i];
}
