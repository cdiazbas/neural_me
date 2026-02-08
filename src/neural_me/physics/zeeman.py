
import torch
from numpy import arange, array, sum

# Lande factor
def g(j, l, s):
    if j == 0:
        return 0.
    return 3. / 2 + (s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


# Effective Lande factor
def geff(ju, gu, jl, gl):
    return 0.5 * (gu + gl) + 0.25 * (gl - gu) * (jl * (jl + 1) - ju * (ju + 1))


# Probability of transition S_{i_j}
def sij(ju, mu, jl, ml):
    fPI = []
    fSR = []
    fSB = []
    if ju - jl == 0.:
        if mu - ml == +1.:  # sigma_b
            s = (ju + mu) * (ju - mu + 1.)
            fSB.append(s)
        if mu - ml == 0.:  # pi
            s = 2. * mu**2.
            fPI.append(s)
        if mu - ml == -1.:  # sigma_r
            s = (ju - mu) * (ju + mu + 1.)
            fSR.append(s)
    if ju - jl == +1.:
        if mu - ml == +1.:  # sigma_b
            s = (ju + mu) * (jl + mu)
            fSB.append(s)
        if mu - ml == 0.:  # pi
            s = 2. * (ju**2. - mu**2.)
            fPI.append(s)
        if mu - ml == -1.:  # sigma_r
            s = (ju - mu) * (jl - mu)
            fSR.append(s)
    if ju - jl == -1.:
        if mu - ml == +1.:  # sigma_b
            s = (jl - mu) * (ju - mu + 2.)
            fSB.append(s)
        if mu - ml == 0.:  # pi
            s = 2. * (jl**2. - mu**2.)
            fPI.append(s)
        if mu - ml == -1.:  # sigma_r
            s = (jl + mu) * (ju + mu + 2.)
            fSR.append(s)
    return fPI, fSR, fSB


# Todas las transiciones posibles
def components(ju, jl):
    PI = []
    SR = []
    SB = []
    # M = -J, ..., +J
    mu = arange(-ju, ju + 1, 1)
    ml = arange(-jl, jl + 1, 1)
    # Index for mu & ml
    lonu = arange(0, len(mu), 1)
    lonl = arange(0, len(ml), 1)
    for i in lonu:
        for j in lonl:
            if mu[i] - ml[j] == 0.:
                PI.append((mu[i], ml[j]))
            if mu[i] - ml[j] == -1.:
                SR.append((mu[i], ml[j]))
            if mu[i] - ml[j] == +1.:
                SB.append((mu[i], ml[j]))
    return [PI, SR, SB]


# Split and strength of transition:
def pattern(ju, lu, su, jl, ll, sl):
    dPI = []
    sPI = []
    dSR = []
    sSR = []
    dSB = []
    sSB = []
    gu = g(ju, lu, su)
    gl = g(jl, ll, sl)
    componentsjujl = components(ju, jl)
    for i in componentsjujl[0]:
        delt = gl * i[1] - gu * i[0]
        dPI.append(delt)
        sPI.append(sij(ju, i[0], jl, i[1])[0])
    for i in componentsjujl[1]:
        delt = gl * i[1] - gu * i[0]
        dSR.append(delt)
        sSR.append(sij(ju, i[0], jl, i[1])[1])
    for i in componentsjujl[2]:
        delt = gl * i[1] - gu * i[0]
        dSB.append(delt)
        sSB.append(sij(ju, i[0], jl, i[1])[2])
    sPI = array(sPI).flatten()
    sSR = array(sSR).flatten()
    sSB = array(sSB).flatten()
    dPI = array(dPI).flatten()
    dSR = array(dSR).flatten()
    dSB = array(dSB).flatten()
    return [[dPI, dSR, dSB], [sPI, sSR, sSB]]

# Notation:
def notation(ju, lu, su, jl, ll, sl):
    lNotation = ['S', 'P', 'D', 'F']
    stringU = r'$^' + str(2 * int(su) + 1) + \
        lNotation[int(lu)] + '_' + str(int(ju)) + '$'
    stringL = r'$^' + str(2 * int(sl) + 1) + \
        lNotation[int(ll)] + '_' + str(int(jl)) + '$'
    return stringL + r'$\rightarrow$' + stringU


def extrae(fichero, lineaToFind):
    fileLineas = open(fichero, 'r')
    for linea in fileLineas:
        if int(linea.split('=')[0]) == lineaToFind:
            datos = linea.split()

    ju = float(datos[11][:-1])
    jl = float(datos[9][:-1])

    lNotation = ['S', 'P', 'D', 'F']
    lu = lNotation.index(datos[10][1])
    ll = lNotation.index(datos[8][1])

    su = (int(datos[10][0]) - 1) / 2
    sl = (int(datos[8][0]) - 1) / 2

    elem = (''.join(datos[2:4]))
    l0 = float(datos[4])
    return [ju, lu, su, jl, ll, sl, elem, l0, datos]


def paramLine(nlinea):

    # Calculating the Zeeman Pattern
    lineaToFind = nlinea
    # TODO: This path needs to be dynamic or configured properly
    # Assuming LINEAS.txt is in the same directory or accessible
    import os
    # Find LINEAS.txt relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lineas_path = os.path.join(current_dir, 'LINEAS.txt')
    
    ju, lu, su, jl, ll, sl, elem, l0, datos = extrae(lineas_path, lineaToFind)
    [dpi, dsr, dsb], [spi, ssr, ssb] = pattern(ju, lu, su, jl, ll, sl)

    # Some useful information
    print('----------------------------------------------------')
    print('Element = {0}'.format(elem))
    print('lambda0 = {0}'.format(l0))
    print('ju={0}, lu={1}, su={2}'.format(ju, lu, su))
    print('jl={0}, ll={1}, sl={2}'.format(jl, ll, sl))

    # g(j,l,s); # geff(ju,gu,jl,gl)
    gu = g(ju, lu, su)
    gl = g(jl, ll, sl)
    gef = geff(ju, gu, jl, gl)
    print('g_u  = {0:2.2f}'.format(gu))
    print('g_l  = {0:2.2f}'.format(gl))
    print('geff = {0:2.2f}'.format(gef))

    # Normalization:
    spi /= sum(spi)
    ssr /= sum(ssr)
    ssb /= sum(ssb)

    return [[torch.tensor(dpi)[None,:], torch.tensor(dsr)[None,:], torch.tensor(dsb)[None,:]], [torch.tensor(spi)[None,:], torch.tensor(ssr)[None,:], torch.tensor(ssb)[None,:]], [ju, lu, su, jl, ll, sl, elem, l0, gu, gl, gef]]

def fvoigt_humlicek(damp, vv):
    """
    Humlicek's 4-region Voigt approximation (matches C++ backend exactly).
    More accurate than Hui's method, especially in line core.
    """
    device = damp.device
    dtype = damp.dtype
    
    # Ensure inputs are complex-compatible
    a = damp.float()
    v = vv.float()
    
    # Compute sav = |v| + a for region selection
    sav = torch.abs(v) + a
    
    # Complex argument: tav = a - i*v
    tav = torch.complex(a, -v)
    uav = tav * tav
    
    # Initialize output
    w4 = torch.zeros_like(tav)
    
    # Region 1: Far wings (sav >= 15)
    mask1 = sav >= 15.0
    if mask1.any():
        w4_1 = tav * 0.5641896 / (0.5 + uav)
        w4 = torch.where(mask1, w4_1, w4)
    
    # Region 2: Intermediate (5.5 <= sav < 15)
    mask2 = (sav >= 5.5) & (sav < 15.0)
    if mask2.any():
        w4_2 = tav * (1.410474 + uav * 0.5641896) / (0.75 + uav * (3.0 + uav))
        w4 = torch.where(mask2, w4_2, w4)
    
    # Region 3: Near core (a >= 0.195*|v| - 0.176 and sav < 5.5)
    mask3 = (a >= (0.195 * torch.abs(v) - 0.176)) & (sav < 5.5)
    if mask3.any():
        # 4th order rational approximation
        num = (16.4955 + tav * (20.20933 + tav * (11.96482 + tav * (3.778987 + tav * 0.5642236))))
        den = (16.4955 + tav * (38.82363 + tav * (39.27121 + tav * (21.69274 + tav * (6.699398 + tav)))))
        w4_3 = num / den
        w4 = torch.where(mask3, w4_3, w4)
    
    # Region 4: Core (default, sav < 5.5 and a < 0.195*|v| - 0.176)
    mask4 = ~(mask1 | mask2 | mask3)
    if mask4.any():
        # 6th order polynomial with exponential
        num = tav * (36183.31 - uav * (3321.9905 - uav * (1540.787 - uav * (219.0313 - uav * (35.76683 - uav * (1.320522 - uav * 0.56419))))))
        den = (32066.6 - uav * (24322.84 - uav * (9022.228 - uav * (2186.181 - uav * (364.2191 - uav * (61.57037 - uav * (1.841439 - uav)))))))
        w4_4 = torch.exp(uav) - num / den
        w4 = torch.where(mask4, w4_4, w4)
    
    # Extract H and F (Faraday-Voigt)
    h = torch.real(w4)
    f = torch.imag(w4) * 0.5
    
    return [h, f]

def fvoigt_hui(damp, vv):
    """
    Hui's single rational approximation (original PyTorch implementation).
    Faster but less accurate than Humlicek's method.
    """
    device = damp.device
    dtype = damp.dtype

    A = torch.tensor([122.607931777104326, 214.382388694706425, 181.928533092181549,
         93.155580458138441, 30.180142196210589, 5.912626209773153,
         0.564189583562615], device=device, dtype=dtype)

    B = torch.tensor([122.60793177387535, 352.730625110963558, 457.334478783897737,
         348.703917719495792, 170.354001821091472, 53.992906912940207,
         10.479857114260399, 1.], device=device, dtype=dtype)

    z = torch.complex(damp.float(), -torch.abs(vv).float())

    Z = ((((((A[6] * z + A[5]) * z + A[4]) * z + A[3]) * z + A[2]) * z + A[1]) * z + A[0]) /\
        (((((((z + B[6]) * z + B[5]) * z + B[4]) * z + B[3]) * z + B[2]) * z + B[1]) * z + B[0])

    h = torch.real(Z)
    f = torch.sign(vv) * torch.imag(Z) * 0.5

    return [h, f]

def fvoigt(damp, vv, method="hui"):
    """
    Compute Voigt and Faraday-Voigt profiles.
    
    Args:
        damp: Damping parameter (a)
        vv: Velocity/frequency parameter (v)
        method: "humlicek" (default, accurate, matches C++) or "hui" (fast)
    
    Returns:
        [H, F]: Voigt profile H and Faraday-Voigt profile F
    """
    if method == "humlicek":
        return fvoigt_humlicek(damp, vv)
    elif method == "hui":
        return fvoigt_hui(damp, vv)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'humlicek' or 'hui'")
