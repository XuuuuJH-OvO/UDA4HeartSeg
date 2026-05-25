input = [4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
level = input[0]
left = level
cnt = level
cur = []
while left <= level * level:
    cur.append(input[left])
    left += level
    if left > level * level:
        left = cnt - 1
        cnt -= 1
    if cnt == 0:
        break
print(','.join(map(str, cur)))