import sys

reset_color = '\033[0m'

colors = [
    # change 3xm to 9xm for high intensity
    '\033[1;31m',
    '\033[1;33m',
    '\033[1;34m',
    '\033[1;35m',
    '\033[1;36m',
    '\033[1;37m',
    '\033[1;32m',
]


def get_color(i):
    i = i % len(colors)
    return colors[i]


if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        data = line.split(' ||| ')
        s1 = data[0].split()
        s2 = data[1].split()

        for i, (alignment, score) in enumerate(zip(data[2].split(), data[3].split())):
            try:
                align = list(map(int, alignment.split('-')))
                print(f'{get_color(i)}{align[0]}:{s1[align[0]]} - {align[1]}:{s2[align[1]]} - {score}{reset_color}')
                s1[align[0]] = f'{get_color(i)}{s1[align[0]]}{reset_color}'
                s2[align[1]] = f'{get_color(i)}{s2[align[1]]}{reset_color}'
            except IndexError:
                print(f'IndexError: {alignment}')

        print(' '.join(s1))
        print(' '.join(s2))
        print()
