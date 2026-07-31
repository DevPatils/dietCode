import random

def bubble_sort(lst):
    n = len(lst)

    for i in range(n):
        # Create a flag that will allow the function to terminate early if there's nothing left to sort
        already_sorted = True

        # Start looking at each item of the list one by one, comparing it with its adjacent value
        for j in range(n - i - 1):
            if lst[j] > lst[j + 1]:
                # Swap values
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                # Since we had to swap two elements, we need to iterate over the list again.
                already_sorted = False

        # If there were no swaps during the last iteration, the list is already sorted, and we can terminate
        if already_sorted:
            break

    return lst

test_list = [64, 34, 25, 12, 22, 11, 90]
print("Original list: ", test_list)
print("Sorted list: ", bubble_sort(test_list))