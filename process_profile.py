import pstats


S = pstats.Stats('profile.txt')

print(S)


S.sort_stats('calls')

S.print_stats(20)
