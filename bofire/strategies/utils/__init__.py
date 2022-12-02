def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)
