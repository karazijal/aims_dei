import io
import requests
import pandas as pd

URL = "http://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/sotonmet.txt"
FILE = 'sotonmet.txt'
LABELS = {
    'time': 'Reading Date and Time (ISO)',
    'tide': 'Tide height (m)',
    'true_tide': 'True tide height (m)',
    'air': 'Air temperature (C)',
    'true_air': 'True air temperature (C)'
}


def sontomet_dataset():
    f = io.StringIO(requests.get(URL).content.decode())
    data = pd.read_csv(f, parse_dates=True)
    t = pd.to_datetime(data[LABELS['time']])
    y = data[LABELS['tide']].to_numpy()
    y_gt = data[LABELS['true_tide']].to_numpy()  # Ground truth

    t = (t - t.min()).apply(lambda x: x.total_seconds() / 3600).to_numpy()

    mask = ~(y != y)
    yt = y[mask]
    xt = t[mask]
    yp = y_gt[~mask]
    xp = t[~mask]
    yg = y_gt

    z = data[LABELS['air']].to_numpy()
    z_gt = data[LABELS['true_air']].to_numpy()  # Ground truth

    mask = ~(z != z)
    zt = z[mask]
    wt = t[mask]
    zp = z_gt[~mask]
    wp = t[~mask]
    zg = z_gt

    return {
        'train_tide': yt,
        'train_tide_t': xt,
        'test_tide': yp,
        'test_tide_t': xp,
        'tide_gt': yg,

        'gt_t': t,

        'train_air': zt,
        'train_air_t': wt,
        'test_air': zp,
        'test_air_t': wp,
        'air_gt': zg
    }


if __name__ == '__main__':
    print(sontomet_dataset())
