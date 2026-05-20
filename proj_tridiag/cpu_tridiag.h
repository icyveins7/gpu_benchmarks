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
