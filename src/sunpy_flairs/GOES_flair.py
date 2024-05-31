from sunpy.net import Fido
from sunpy.net import attrs as a


event_type = "FL"
tstart = "2011/10/01"
tend = "2019/12/31"

result = Fido.search(a.Time(tstart, tend),
                     a.hek.EventType(event_type),
                     a.hek.FL.GOESCls > "B10.0",
                     a.hek.FL.GOESCls < "X10.0",
                     a.hek.Event.Coord1 < 800,
                     a.hek.OBS.Observatory == "GOES")


# Here we only show two columns due there being over 100 columns returned normally.
# print(result.show("hpc_bbox", "refs"))

# It"s also possible to access the HEK results from the
# `~sunpy.net.fido_factory.UnifiedResponse` by name.
hek_results = result["hek"]

# keywords
#print(hek_results.colnames[::10])
# print(result["hek"]["fl_peakflux"])

filtered_results = hek_results["event_starttime", "event_peaktime",
                               "event_endtime", "fl_goescls", "ar_noaanum"]


by_magnitude = sorted(filtered_results, key=lambda x: ord(x['fl_goescls'][0]) + float(x['fl_goescls'][1:]), reverse=True)

#for flare in by_magnitude:
#    print(flare['fl_goescls'], flare['event_starttime'])

# filtered_results.write("C1_flares_SC24.txt", format="ascii")
