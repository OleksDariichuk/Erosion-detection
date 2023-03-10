def sum_up_to_n(n):
    return (n * (n + 1)) // 2 


n = int(input("Enter a positive integer: "))

print(f"The sum of integers between 1 and {n} is {sum_up_to_n(n)}")