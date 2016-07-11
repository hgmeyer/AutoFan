def lowpass(x, y_old, rc, dt):
    '''
    1st order lowpass filter
    '''

    alpha = float(dt) / (rc + dt)
    return alpha * x + (1 - alpha) * y_old


def highpass(x, x_old, y_old, rc, dt):
    '''
    1st order highpass filter
    '''

    alpha = float(dt) / (rc + dt)
    return alpha * y_old + alpha(x - x_old)
