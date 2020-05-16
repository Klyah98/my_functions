def subtract(string, position):
    subtraction = int(string[position]) - 1
    return string[:position] + str(subtraction) + string[position + 1:]


def subtract_one_augmentation(string):
    aug = []
    string_indices = range(len(string) - 1, -1, -1)
    for i in string_indices:
        if (string[i] != 'X') and (string[i] != '0'):
            aug.append(subtract(string, i))
    return aug


def one_zero_augmentation(string):
    if string[-1] == '0':
        previous_was_zero = True
    else:
        previous_was_zero = False
    aug = []
    string_indices = range(len(string) - 1, -1, -1)
    for i in string_indices:
        if string[i] == '0':
            previous_was_zero = True
        else:
            if previous_was_zero:
                aug.append(string[:i + 1] + '0' + string[i + 1:])
                previous_was_zero = False
    if previous_was_zero:
        aug.append('0' + string)
    return aug



def two_zero_augmentation(string):
    if string[-1] == '0':
        previous_was_zero = True
    else:
        previous_was_zero = False
    aug = []
    string_indices = range(len(string) - 1, -1, -1)
    for i in string_indices:
        if string[i] == '0':
            previous_was_zero = True
        else:
            if previous_was_zero:
                aug.append(string[:i + 1] + '00' + string[i + 1:])
                previous_was_zero = False
    if previous_was_zero:
        aug.append('00' + string)
    return aug


def subtract_two_augmentation(string):
    aug = []
    string_indices = range(len(string) - 1, -1, -1)
    for i in string_indices:
        if (string[i] != 'X') and (string[i] != '0'):
            one_subtract_str = subtract(string, i)
            for j in range(i - 1, -1, -1):
                if (one_subtract_str[j] != 'X') and (one_subtract_str[j] != '0'):
                    aug.append(subtract(one_subtract_str, j))
    return aug


def positive_augmentation(string, num_of_aug):
    modified_strings = []
    subtract_one_aug = subtract_one_augmentation(string)
    modified_strings += subtract_one_aug
    for one_aug_str in subtract_one_aug:
        modified_strings += one_zero_augmentation(one_aug_str)
        modified_strings += two_zero_augmentation(one_aug_str)
    subtract_two_aug = subtract_two_augmentation(string)
    modified_strings += subtract_two_aug
    for two_aug_str in subtract_two_aug:
        modified_strings += one_zero_augmentation(two_aug_str)
        modified_strings += two_zero_augmentation(two_aug_str)
    if num_of_aug <= len(modified_strings):
        return modified_strings[:num_of_aug]
    else:
        return modified_strings*(num_of_aug//len(modified_strings)) + modified_strings[:(num_of_aug%len(modified_strings))]


sample1 = '111102122221221111111021111011110332002444321111011354321221102100'
sample2 = '111102122221221111X111021111011110332002444X321111011354321221102100'

print(sample2)
print('\n'.join(positive_augmentation(sample2, 10)))
