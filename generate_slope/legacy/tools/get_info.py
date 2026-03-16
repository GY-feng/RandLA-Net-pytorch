import laspy

las = laspy.read(r"D:\CloudPointProcessing\PCGSPRO_1761030020\wappe2007@qq.com\DJI_202512161455_337_惠清汕头方向-K309-036-159-点云\lidars\terra_las\cloud_merged.las")

print("=== 点云包含的字段 ===")
for dim in las.point_format.dimensions:
    print(f"- {dim.name} ({dim.dtype})")
