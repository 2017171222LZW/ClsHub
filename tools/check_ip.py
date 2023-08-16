'''
杀死外来ip进程pid: 2201
'''
import IPy
import time
import random
import hashlib
import argparse
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
}


def getaliIP(ip):
    # 这里使用ali的IP地址查询功能。
    # https://market.aliyun.com/products/?keywords=ip%E5%BD%92%E5%B1%9E%E5%9C%B0
    # 需要自己去这个网址注册账号，然后进行调用。
    # 这里我们先进行定义url
    host = 'https://ips.market.alicloudapi.com'
    path = '/iplocaltion'
    method = "GET"
    appcode = '填写你自己的xxx'
    url = host + path + '?ip=' + ip
    # 定义头部。
    headers = {"Authorization": 'APPCODE ' + appcode}
    try:
        # 进行获取调用结果。
        rep = requests.get(url, headers=headers)
    except:
        return 'url参数错误'
    # 判断是否调用成功。如果调用成功就接着进行下边的动作。
    httpStatusCode = rep.status_code
    if httpStatusCode == 200:
        # 转换成json格式
        data = rep.json()
        # 然后获取其中的参数。
        ''''
        # 是以下边这种格式进行返回的。
        {
          "code": 100,
          "message": "success",
          "ip": "110.188.234.66",
          "result": {
               "en_short": "CN", // 英文简称
          "en_name": "China", // 归属国家英文名称
        "nation": "中国", // 归属国家
        "province": "四川省", // 归属省份
        "city": "绵阳市", // 归属城市
        "district": "涪城区", // 归属县区
        "adcode": 510703, // 归属地编码
        "lat": 31.45498, // 经度
        "lng": 104.75708 // 维度
        }
        }'''
        result1 = data.get('result')
        city = result1['city']
        province = result1['province']
        nation = result1['nation']
        district = result1['district']
        latitude = result1['lat']
        longitude = result1['lng']
        # 返回我们需要的结果。
        result = '-' * 50 + '\n' + \
                 '''[ali.com查询结果-IP]: %s\n经纬度: (%s, %s)\n国家: %s\n地区: %s\n城市: %s\n''' % (
                     ip, longitude, latitude, nation, province, city) \
                 + '-' * 50
    else:
        httpReason = rep.headers['X-Ca-Error-Message']
        if (httpStatusCode == 400 and httpReason == 'Invalid Param Location'):
            return "参数错误"
        elif (httpStatusCode == 400 and httpReason == 'Invalid AppCode'):
            return "AppCode错误"
        elif (httpStatusCode == 400 and httpReason == 'Invalid Url'):
            return "请求的 Method、Path 或者环境错误"
        elif (httpStatusCode == 403 and httpReason == 'Unauthorized'):
            return "服务未被授权（或URL和Path不正确）"
        elif (httpStatusCode == 403 and httpReason == 'Quota Exhausted'):
            return "套餐包次数用完"
        elif (httpStatusCode == 500):
            return "API网关错误"
        else:
            return "参数名错误 或 其他错误" + httpStatusCode + httpReason

    return result


'''淘宝API'''


def getTaobaoIP(ip):
    # 请求淘宝获取IP位置的API接口，但是现在有些不是很好用了。查不出来了。
    # 看了看接口需要进行传入秘钥
    url = 'http(s)://ips.market.alicloudapi.com/iplocaltion'
    # 使用get方法进行请求。
    res = requests.get(url + ip, headers=headers)
    # 然后进行解析参数。
    data = res.json().get('data')
    print(res.json)
    if data is None:
        return '[淘宝API查询结果-IP]: %s\n无效IP' % ip
    result = '-' * 50 + '\n' + \
             '''[淘宝API查询结果-IP]: %s\n国家: %s\n地区: %s\n城市: %s\n''' % (
             ip, data.get('country'), data.get('region'), data.get('city')) \
             + '-' * 50
    return result


'''ip-api.com(很不准)'''


def getIpapiIP(ip):
    url = 'http://ip-api.com/json/'
    res = requests.get(url + ip, headers=headers)
    data = res.json()
    yd = youdao()
    city = yd.translate(data.get('city'))[0][0]['tgt']
    country = yd.translate(data.get('country'))[0][0]['tgt']
    region_name = yd.translate(data.get('regionName'))[0][0]['tgt']
    latitude = data.get('lat')
    longitude = data.get('lon')
    result = '-' * 50 + '\n' + \
             '''[ip-api.com查询结果-IP]: %s\n经纬度: (%s, %s)\n国家: %s\n地区: %s\n城市: %s\n''' % (
             ip, longitude, latitude, country, region_name, city) \
             + '-' * 50
    return result


'''ipstack.com'''


def getIpstackIP(ip):
    # 定义url
    url = 'http://api.ipstack.com/{}?access_key=1bdea4d0bf1c3bf35c4ba9456a357ce3'
    res = requests.get(url.format(ip), headers=headers)
    data = res.json()
    # 实例化一个有道翻译的类。
    yd = youdao()
    # 调用翻译函数。获取翻译的值。
    continent_name = yd.translate(data.get('continent_name'))[0][0]['tgt']
    country_name = yd.translate(data.get('country_name'))[0][0]['tgt']
    region_name = yd.translate(data.get('region_name'))[0][0]['tgt']
    city = yd.translate(data.get('city'))[0][0]['tgt']
    # 获取经纬度。
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    result = '-' * 50 + '\n' + \
             '''[ipstack.com查询结果-IP]: %s\n经纬度: (%s, %s)\n板块: %s\n国家: %s\n地区: %s\n城市: %s\n''' % (
             ip, longitude, latitude, continent_name, country_name, region_name, city) \
             + '-' * 50
    return result


'''IP地址有效性验证'''


def isIP(ip):
    try:
        IPy.IP(ip)
        return True
    except:
        return False


'''
Function:
  有道翻译类，进行翻译上边我们查询结果的返回值。
'''


class youdao():
    def __init__(self):
        # 这里我们需要使用post方法进行调用接口。
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }

        self.data = {
            'i': 'hello',
            'action': 'FY_BY_CLICKBUTTION',
            'bv': 'e2a78ed30c66e16a857c5b6486a1d326',
            'client': 'fanyideskweb',
            'doctype': 'json',
            'from': 'AUTO',
            'keyfrom': 'fanyi.web',
            'salt': '15532627491296',
            'sign': 'ee5b85b35c221d9be7437297600c66df',
            'smartresult': 'dict',
            'to': 'AUTO',
            'ts': '1553262749129',
            'typoResult': 'false',
            'version': '2.1'
        }
        # 有道翻译调用接口的url
        self.url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
        # http: // fanyi.youdao.com / translate?smartresult = dict & smartresult = rule & sessionFrom =

    # 进行翻译。
    def translate(self, word):
        # 先判断单词是否为空。
        if word is None:
            return [word]
        # 随机生成一个时间。
        t = str(time.time() * 1000 + random.randint(1, 10))
        t = str(time.time() * 1000 + random.randint(1, 10))
        # 传入我们需要翻译的单词和其他参数。
        self.data['i'] = word
        self.data['salt'] = t
        sign = 'fanyideskweb' + word + t + '6x(ZHw]mwzX#u0V7@yfwK'
        # 这里需要哈希一下。
        self.data['sign'] = hashlib.md5(sign.encode('utf-8')).hexdigest()
        # 进行post方法调用接口，并获取我们需要的参数。
        res = requests.post(self.url, headers=self.headers, data=self.data)
        # 返回翻译的结果。
        return res.json()['translateResult']


'''主函数'''


def main(ip):
    separator = '*' * 30 + 'IPLocQuery' + '*' * 30
    # 首先判断IP地址是否合法。
    if isIP(ip):
        # 然后分别调用几个接口进行查询。
        print(separator)
        print(getaliIP(ip))
        print(getIpstackIP(ip))
        print(getIpapiIP(ip))
        print('*' * len(separator))
    else:
        print(separator + '\n[Error]: %s --> 无效IP地址...\n' % ip + '*' * len(separator))


if __name__ == '__main__':
    # 获取终端输入的入参。
    parser = argparse.ArgumentParser(description="Query geographic information based on IP address.")
    # 可选参数，代表着文件的名字，里边存放着IP之地。
    parser.add_argument('-f', dest='filename', help='File to be queried with one ip address per line')
    # 可选参数，代表着我们需要查询的IP地址。
    parser.add_argument('-ip', dest='ipaddress', help='Single ip address to be queried')
    args = parser.parse_args()
    # 获取终端输入的参数。
    ip = args.ipaddress
    filename = args.filename
    # 判断终端是否有进行输入参数。
    if ip:
        main(ip)
    if filename:
        with open(filename) as f:
            # 获取文件中的所有IP地址，存放成一个列表的形式。
            ips = [ip.strip('\n') for ip in f.readlines()]
        for ip in ips:
            main(ip)


