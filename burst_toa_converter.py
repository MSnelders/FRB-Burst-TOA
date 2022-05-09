# aaa

import numpy as np
import pandas as pd
import argparse
import os
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy import coordinates as coord


def options():
    parser = argparse.ArgumentParser(description="Python program that converts topocentric burst times at a reference frequency to barycentric times at infinite frequency. For the most update-to-date locations of radio telescopes see the file 'locations.dat' on the github page of pySCHED: https://github.com/jive-vlbi/sched")
    general = parser.add_argument_group('Arguments')
    general.add_argument('-MJD_start', type=str, default='56789.1234567',
                         help="Start of the dataset(s) in MJD. If multiple starts are given, seperate them with a ','. Default: 56789.1234567")
    general.add_argument('-MJD_scale', type=str, default='utc',
                         help="Timescale of the start of the dataset(s), Default: 'utc'")
    general.add_argument('-burst_tstamps', type=str, default='0.0,12.345,456.789',
                         help="Seconds into the file the burst(s) occur. If multiple timestamps are given, seperate them with a ','. Default: '0.0,12.345,456.789'")
    general.add_argument('-source_name', type=str, default='SGR1935+2154',
                         help="Name of the source (for printing purposes only). Default: 'SGR1935+2154'")
    general.add_argument('-ra', type=str, default='19h34m55.680s',
                         help="The RA (right ascension) of the source. The format is 'xxhyymzz.zzs'. Default: '19h34m55.680s'")
    general.add_argument('-dec', type=str, default='+21d53m48.20s',
                         help="The DEC (Declination) of the source. The format is '+-xxdyymzz.zzs'. Default: '+21d53m48.20s'")
    general.add_argument('-station_name', type=str, default='Effelsberg',
                         help="Name of the staion (for printing purposes only). Default: 'Effelsberg'")
    general.add_argument('-X', type=np.float64, default=4033947.23550,
                         help="Geocentric X coordinate of the station in meters. Default: '4033947.23550'")
    general.add_argument('-Y', type=np.float64, default=486990.79430,
                         help="Geocentric Y coordinate of the station in meters. Default: '486990.79430'")
    general.add_argument('-Z', type=np.float64, default=4900431.00170,
                         help="Geocentric Z coordinate of the station in meters. Default: '4900431.00170'")
    general.add_argument('-DM', type=np.float64, default=332.7206,
                         help="The dispersion measure in units of pc cm^-3. Default: '332.7206'")
    general.add_argument('-ref_freq', type=np.float64, default=1400.0,
                         help="The reference frequency in units of MHz. Default: '1400'")
    general.add_argument('-DM_constant', type=np.float64, default=4.1488064239,
                         help="The dispersion constant in units of GHz^2 cm^3 pc^-1 ms. Default: '4.1488064239' (see also https://arxiv.org/pdf/2007.02886.pdf)")
    general.add_argument('-pandas-sig', type=int, default=16,
                         help="Number of significant digits to print of the pandas DataFrame. Default: '16'.")
    general.add_argument('-save', type=bool, default=False,
                         help="Boolean flag. If true will save the pandas DataFrame as a .pkl file. Default: 'False'.")
    general.add_argument('-outname', type=str, default='burst_times.pkl',
                         help="Filename to use if the pandas DataFrame gets saved. Default: 'burst_times.pkl'.")
    return parser.parse_args()


def calc_time_delay(freq, dm, dmconst):
    """
    Input:
        freq (MHz), float
        dm (pc/cc), float
        dmconst (GHz^2 cm^3 pc^-1 ms), float
    Returns:
        The time difference between infinite frequency and freq
        in SECONDS"""
    return dmconst * 10**6. * dm * freq**-2. / 1000.


def print_input(args, MJDs, burst_tstamps):
    print("-------------------------------------")
    print("This program comes with absolutely no warrenty.")
    print("You have used the following input values:")
    print(f"Station: {args.station_name}, X = {args.X} m, Y = {args.Y} m, Z = {args.Z} m")
    print(f"Source: {args.source_name}, ra = {args.ra}, DEC = {args.dec}")
    print(f"DM = {args.DM} pc/cc, DM_constant = {args.DM_constant} GHz^2 cm^3 pc^-1 ms")
    print(f"Reference frequency = {args.ref_freq} MHz")
    print(f"MJD-input timescale: {args.MJD_scale}")
    print(f"Start of the dataset(s): {MJDs}")
    print(f"Burst timestamps (seconds): {burst_tstamps}")
    print("-------------------------------------")
    return

def main(args):
    # extract MJDs and timestamps from string and convert to floats
    MJDs = [np.float64(x) for x in args.MJD_start.split(",")]
    burst_tstamps = [np.float64(x) for x in args.burst_tstamps.split(",")]

    if len(burst_tstamps) > 1:
        # check for valid inputs
        if not ((len(MJDs) == 1) or (len(MJDs) == len(burst_tstamps))):
            raise ValueError("if N > 1 burst_tstamps are given, MJD_start must contain either 1 or N entries.")
        # expand MJDs if needed
        if len(MJDs) == 1:
            MJDs = MJDs * len(burst_tstamps)

    # initialize source and observing station
    source = coord.SkyCoord(args.ra, args.dec, unit = (u.hourangle, u.deg))
    station = EarthLocation.from_geocentric(x = args.X * u.m,\
                                            y = args.Y * u.m,\
                                            z = args.Z * u.m)

    # set number of significant digits to print at the end
    pd.set_option("display.precision", args.pandas_sig)

    # initialize DataFrame
    df = pd.DataFrame(data={"MJD_start":MJDs,\
                            "tstamp_ref_freq_sec":burst_tstamps})

    # calculate topocentric burst TOA at the reference frequency in utc scale
    df["mjd_topo_ref_freq_utc"] = (Time(df["MJD_start"], format='mjd',\
            scale=args.MJD_scale, location=station) +\
            df["tstamp_ref_freq_sec"].values * u.second).utc.value

    # calculate topocentric burst TOA at infinite frequency in utc scale
    df["mjd_topo_inf_freq_utc"] = (Time(df["mjd_topo_ref_freq_utc"],\
            format='mjd', scale='utc', location=station) -\
            calc_time_delay(args.ref_freq, args.DM, args.DM_constant) * u.second).utc.value

    # calculate barycentric burst TOA at infinite frequency in utc scale
    df["mjd_bary_inf_freq_utc"] = (Time(df["mjd_topo_inf_freq_utc"], format='mjd',\
        location=station, scale='utc') +\
        Time(df["mjd_topo_inf_freq_utc"], format='mjd',\
            location=station, scale='utc').light_travel_time(source, 'barycentric')).utc.value

    # calculate barycentric burst TOA at infinite frequency in tdb scale
    df["mjd_bary_inf_freq_tdb"] = (Time(df["mjd_topo_inf_freq_utc"], format='mjd',\
        location=station, scale='utc') +\
        Time(df["mjd_topo_inf_freq_utc"], format='mjd',\
            location=station, scale='utc').light_travel_time(source, 'barycentric')).tdb.value

    # print input values from argparse
    print_input(args, MJDs, burst_tstamps)
    # print the dirived
    pd.set_option("display.max_colwidth", args.pandas_sig * len(df.columns))
    pd.set_option("display.max_rows", len(burst_tstamps))

    if args.save:
        if os.path.exists(args.outname):
            print(f"WARNING: Not saving the file. {args.outname} already exists.")
        else:
            print(f"Saving DataFrame to {args.outname}")
            df.to_pickle(args.outname)
        print("-------------------------------------")

    print(df)
    return


if __name__ == "__main__":
    args = options()
    main(args)
