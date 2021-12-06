import time

if __name__ == '__main__':
    x = input("insert WITHOUT STARTING AND ENDING BRACES!\n")
    y = x.split("}, {")
    z = [float(a) for a in y]

    a = input("insert WITHOUT STARTING AND ENDING BRACES!\n")
    b = a.split(", ")
    c = [float(d) for d in b]
    p=1
    for i in range(len(z)):
        v = z[i]
        s = -c[i]
        t = 0.001
        print(s)
        s = -c[i]+v * t - 0.5 * 9.81 * t * t
        while s > -c[i]:
            print(s)
            s = -c[i]+(v * t - 0.5 * 9.81 * t * t)
            t+=0.001
