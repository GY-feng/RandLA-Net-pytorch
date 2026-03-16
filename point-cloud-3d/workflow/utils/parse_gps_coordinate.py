from fractions import Fraction

def parse_gps_coordinate(coord: str, ref: str):
    """
    解析度分秒格式GPS坐标为十进制度
    
    参数:
        coord: EXIF格式的坐标数组，如[degrees, minutes, seconds]
        ref: 方向参考('N','S','E','W')
        
    返回:
        十进制度坐标值

    样例:
        >>> parse_gps_coordinate([40, 59, 36.0], 'N')
        40.98333333333333
    """
    # 解析度、分、秒
    coord = eval(coord)
    degrees = float(coord[0])
    minutes = float(coord[1])
    
    # 处理秒数(可能是分数形式)
    if isinstance(coord[2], (tuple, list)):  # 有些EXIF存储为分数形式
        seconds = Fraction(coord[2][0], coord[2][1])
    elif '/' in str(coord[2]):  # 分数形式如"84121/2000"
        numerator, denominator = map(int, str(coord[2]).split('/'))
        seconds = Fraction(numerator, denominator)
    else:
        seconds = float(coord[2])
    
    # 计算十进制度
    decimal_degrees = degrees + minutes/60 + float(seconds)/3600
    
    # 根据方向参考调整正负
    if ref in ['S', 'W']:
        decimal_degrees = -decimal_degrees
    return decimal_degrees