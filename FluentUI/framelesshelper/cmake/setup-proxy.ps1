# MIT License
#
# Copyright (C) 2023 by wangwenx190 (Yuhang Zhao)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

$__proxy_enable = Get-ItemPropertyValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" "ProxyEnable"
$__proxy_server = Get-ItemPropertyValue "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" "ProxyServer"
if ($__proxy_enable -eq 1) {
    $env:HTTP_PROXY="http://$__proxy_server"
    $env:HTTPS_PROXY="http://$__proxy_server"
    $env:ALL_PROXY="http://$__proxy_server"
} else {
    $env:HTTP_PROXY=""
    $env:HTTPS_PROXY=""
    $env:ALL_PROXY=""
}
Write-Host "HTTP_PROXY:" $env:HTTP_PROXY
Write-Host "HTTPS_PROXY:" $env:HTTPS_PROXY
Write-Host "ALL_PROXY:" $env:ALL_PROXY
