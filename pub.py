# if True:  # dev
if False:  # dev
    # if False:  # dev
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = SHc.C[:, 0]
    ys = SHc.C[:, 1]
    zs = SHc.F
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')
    # ax.set_zlim3d(0.49, 0.51)

    plt.show()

if False:  # dev
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = SHc.C[:, 0]
    # X = numpy.sort(SHGO.C[:, 0])
    X = numpy.linspace(SHc.bounds[0][0], SHc.bounds[0][1])
    Y = SHc.C[:, 1]
    # Y = numpy.sort(SHGO.C[:, 1])
    Y = numpy.linspace(SHc.bounds[1][0], SHc.bounds[1][1])
    xg, yg = numpy.meshgrid(X, Y)
    Z = numpy.zeros((xg.shape[0],
                     yg.shape[0]))

    for i in range(xg.shape[0]):
        for j in range(yg.shape[0]):
            # Z[i, j] = SHGO.F[i]
            Z[i, j] = SHc.func([xg[i, j], yg[i, j]])

    # =Z = SHGO.F

    if True:
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0,
                        antialiased=True, alpha=1.0, shade=True)
        if False:
            cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F')
    # plt.show()

    plt.figure()
    cs = plt.contour(xg, yg, Z, cmap='binary_r')
    plt.clabel(cs)

if False:
    def find_neighbors(pindex, triang):
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:
               triang.vertex_neighbor_vertices[0][pindex + 1]]


    from scipy.spatial import Delaunay

    points = SHc.C
    tri = Delaunay(points)
    if True:
        import matplotlib.pyplot as plt

        plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
        plt.plot(points[:, 0], points[:, 1], 'o')

    if True:
        print('SHGO.C = {}'.format(SHc.C))
        print('tri.points = {}'.format(tri.points))

        print('tri.simplices = {}'.format(tri.simplices))

        # print('numpy.sort(tri.simplices, axis=0)'
        #      ' = {}'.format(numpy.sort(tri.simplices, axis=0)))
        # print('points[tri.simplices] = {}'.format(points[tri.simplices]))
        # print('tri.neighbors[1] = {}'.format(tri.neighbors[1]))
        # print('tri.vertex_neighbor_vertices '
        #       '= {}'.format(tri.vertex_neighbor_vertices))
        # print('tri.vertex_neighbor_vertices[0] '
        #       '= {}'.format(tri.vertex_neighbor_vertices[0]))
        # print('Tuple of two ndarrays of int: (indices, indptr).'
        #       ' The indices of neighboring vertices of vertex k are '
        #       'indptr[indices[k]:indices[k+1]].')
        #
        # print('tri.vertex_to_simplex[0] '
        #       '= {}'.format(tri.vertex_to_simplex[0]))

        print('tri.find_simplex(SHGO.C[0])'
              ' = {}'.format(tri.find_simplex(SHc.C[0])))

        print('tri.simplices[tri.find_simplex(SHGO.C[1])]'
              ' = {}'.format(tri.simplices[tri.find_simplex(SHc.C[1])]))

        print('=' * 10)

        # neighbor_indices = find_neighbors(0, tri)
        neighbor_indices = find_neighbors(0, tri)
        print('neighbor_indices = find_neighbors(0, tri)'
              ' = {}'.format(neighbor_indices))
        print('points[neighbor_indices] = '
              '{}'.format(points[neighbor_indices]))
        print('SHGO.C [neighbor_indices] '
              '= {}'.format(SHc.C[neighbor_indices]))

    if False:
        plt.show()