from collections import defaultdict

import numpy as np
import pandas as pd


from augmentation_final import get_augmentations_old

augmentation_dict = {1: [2, 1, 0, 'change_zero'],
                     2: [3, 2, 1],
                     3: [4, 3, 2],
                     4: [5, 4, 3],
                     5: [5, 4]}




def get_rid_of_X(string):
    return [int(i) for i in string if i != 'X']


def transform_string(string):
    return ''.join([i for i in string if i != 'X'])


def get_prob_dict(string):
    array1 = get_rid_of_X(string)
    array2 = [max(0, i - 1) for i in array1]
    array3 = [max(0, i - 2) for i in array1]
    array4 = [max(0, i - 3) for i in array1]
    array5 = [max(0, i - 4) for i in array1]
    prob1 = sum(array1) / (sum(array1) + len(array1))
    prob2 = sum(array2) / (sum(array2) + len(array2))
    prob3 = sum(array3) / (sum(array3) + len(array3))
    prob4 = sum(array4) / (sum(array4) + len(array4))
    prob5 = sum(array5) / (sum(array5) + len(array5))
    prob_dict = {i + 1: x for i, x in enumerate([prob1, prob2, prob3, prob4, prob5])}
    for key, element in prob_dict.items():
        if key == 1:
            continue
        else:
            if (prob_dict[key - 1] != 0) and (prob_dict[key] == 0):
                prob_dict[key] = prob_dict[key - 1] / 10
    return prob_dict


def get_prob_dicts(string, num_of_dicts):
    result = dict()
    result[1] = get_prob_dict(string)
    for i in range(2, 1 + num_of_dicts):
        prob_dict_ = {key: value ** i for key, value in result[1].items()}
        result[i] = prob_dict_
    return result


def get_augmentation_dict(prob_dict, coef):
    return {key: int(coef * value) for key, value in prob_dict.items()}




def insert_element(ppt_string, element, position):
    return ppt_string[:position] + str(element) + ppt_string[position:]


def augmentation_1(ppt_string, element_to_add: 'int - from 1 to 5', priority_array, num_of_aug):
    """
    TODO comment params!

    :param ppt_string:
    :param element_to_add:
    :param priority_array:
    :param num_of_aug:
    
    :return:
    """
    augmentation = []
    count = 0
    label = True
    while (count < num_of_aug) and label:
        for priority_element in priority_array:
            for i in range(len(ppt_string) - 1, -1, -1):
                if ppt_string[i] == 'X':
                    continue
                if (priority_element == 'change_zero') and (int(ppt_string[i]) == 0):
                    aug = ppt_string[:i] + str(element_to_add) + ppt_string[i + 1:]
                    augmentation.append(aug)
                    count += 1
                elif int(ppt_string[i]) == priority_element:
                    if element_to_add <= priority_element:
                        aug = insert_element(
                            ppt_string=ppt_string,
                            element=element_to_add,
                            position=i + 1
                        )
                    else:
                        aug = insert_element(
                            ppt_string=ppt_string,
                            element=element_to_add,
                            position=i
                        )
                    augmentation.append(aug)
                    count += 1
                if count == num_of_aug:
                    break
            if count == num_of_aug:
                break
        if count == 0:
            label = False
    return augmentation


def augmentation_2(ppt_string, element_to_add, priority_array, num_of_aug):
    first_aug_str = augmentation_1(ppt_string, element_to_add, priority_array, 1)[0]
    return augmentation_1(first_aug_str, element_to_add, priority_array, num_of_aug)


def augmentation_3(ppt_string, element_to_add, priority_array, num_of_aug):
    first_aug_str = augmentation_1(ppt_string, element_to_add, priority_array, 2)[1]
    second_aug_str = augmentation_1(first_aug_str, element_to_add, priority_array, 1)[0]
    return augmentation_1(second_aug_str, element_to_add, priority_array, num_of_aug)


def augmentation_4(ppt_string, element_to_add, priority_array, num_of_aug):
    first_aug_str = augmentation_1(ppt_string, element_to_add, priority_array, 3)[2]
    second_aug_str = augmentation_1(first_aug_str, element_to_add, priority_array, 2)[1]
    third_aug_str = augmentation_1(second_aug_str, element_to_add, priority_array, 1)[0]
    return augmentation_1(third_aug_str, element_to_add, priority_array, num_of_aug)


def augmentation_5(ppt_string, element_to_add, priority_array, num_of_aug):
    first_aug_str = augmentation_1(ppt_string, element_to_add, priority_array, 4)[3]
    second_aug_str = augmentation_1(first_aug_str, element_to_add, priority_array, 3)[2]
    third_aug_str = augmentation_1(second_aug_str, element_to_add, priority_array, 2)[1]
    fourth_aug_str = augmentation_1(third_aug_str, element_to_add, priority_array, 1)[0]
    return augmentation_1(fourth_aug_str, element_to_add, priority_array, num_of_aug)


AUG_FUNCS = {
    1: augmentation_1,
    2: augmentation_2,
    3: augmentation_3,
    4: augmentation_4,
    5: augmentation_5
}


def get_augmentations(ppt_string, coef_):
    ever_eldest_ppt_element = 5
    num_symbols_step = 5
    max_num_elements_to_add = 5
    short_string_len = len(transform_string(ppt_string))
    current_ppt_eldest_element = max(get_rid_of_X(ppt_string))
    eldest_element_to_add = min(ever_eldest_ppt_element, current_ppt_eldest_element + 1)

    prob_dicts = get_prob_dicts(ppt_string, max_num_elements_to_add)
    aug_dicts = dict()
    for i in range(max_num_elements_to_add):
        aug_dicts[i + 1] = get_augmentation_dict(prob_dicts[i + 1], coef_)

    # We make augmentations only for strings of length, greater than border
    if short_string_len <= num_symbols_step:
        return None

    for current_max_num_elements_to_add_ in range(max_num_elements_to_add):
        current_max_num_elements_to_add = current_max_num_elements_to_add_ + 1
        if short_string_len > (num_symbols_step * max_num_elements_to_add):
            list_of_num_elements_to_add = list(range(1, max_num_elements_to_add + 1))
        elif short_string_len <= (num_symbols_step * (current_max_num_elements_to_add + 1)):
            list_of_num_elements_to_add = list(range(1, current_max_num_elements_to_add + 1))
        else:
            continue

        modified_strings = list()
        for num_elements_to_add in list_of_num_elements_to_add:
            for element_to_add in range(1, eldest_element_to_add + 1):
                if aug_dicts[num_elements_to_add][element_to_add] != 0:
                    modified_strings += AUG_FUNCS[num_elements_to_add](
                        ppt_string=ppt_string,
                        element_to_add=element_to_add,
                        priority_array=augmentation_dict[element_to_add],
                        num_of_aug=aug_dicts[num_elements_to_add][element_to_add]
                    )
        return modified_strings


def main():
    """
        Небольшй тест. Имеем 2 строки, отличающиеся только наличием икса в sample1.
        Сделаем аугментации sample, sample1 и удалим из аугментаций sample1 иксы. Поскольку положение иксов не влияет
        на аугментации, должны полочить в точности аугментации строки sample
    """

    sample0 = '111102122221221111111021111011110332002444321111011354321221102100'
    sample1 = '11110212222122111111102111101111033200244432X1111011354321221102100'

    # слева - аугментации sample1, из которых удалены иксы, справа - аугментации sample
    test_res = [transform_string(i) for i in augmentation_1(sample1, 2, augmentation_dict[2], 10)] \
               == augmentation_1(sample0, 2, augmentation_dict[2], 10)
    print(f'Test passed: {test_res}')


    for i, paymnt_pat in enumerate([
        '110212121100021001011000332002104321100011054321021002100',
        '111102122221221111111021111011110332002444321111011354321221102100',
        '5XXX0XXX5XXX0XXX5XXX0',
        '0000032100010000011000021000110000321100'
    ]):
        print(f'Test 1:\n'
              f'{paymnt_pat}\n'
              f'----------------------------')
        augmentations = get_augmentations(paymnt_pat, 100)
        print('\n'.join(augmentations))

    print(f'Final test passed:\n'
          f'{get_augmentations("000003210001000", 100) == get_augmentations_old("000003210001000", 100)}')


if __name__ == '__main__':
    main()

