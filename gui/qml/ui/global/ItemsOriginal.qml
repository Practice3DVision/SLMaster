pragma Singleton

import QtQuick 2.15
import FluentUI 1.0

FluObject{

    property var navigationView

    function rename(item, newName){
        if(newName && newName.trim().length>0){
            item.title = newName;
        }
    }

    FluPaneItem{
        id:item_device
        url: "qrc:/ui/page/Page_Device.qml"
        onTap:{
            if(navigationView.getCurrentUrl()){
                item_device.count = 0
            }
            navigationView.push(url)
        }
    }

    FluPaneItem{
        id:item_calibration
        url: "qrc:/ui/page/Page_Calibration.qml"
        onTap:{
            if(navigationView.getCurrentUrl()){
                item_calibration.count = 0
            }
            navigationView.push(url)
        }
    }

    FluPaneItem{
        id:item_scan_mode
        url: "qrc:/ui/page/Page_ScanMode.qml"
        onTap:{
            if(navigationView.getCurrentUrl()){
                item_scan_mode.count = 0
            }
            navigationView.push(url)
        }
    }

    FluPaneItem{
        id:item_scan
        url: "qrc:/ui/page/Page_Scan.qml"
        onTap:{
            if(navigationView.getCurrentUrl()){
                item_scan.count = 0
            }
            navigationView.push(url)
        }
    }

    FluPaneItem{
        id:item_post_process
        url: "qrc:/ui/page/Page_PostProcess.qml"
        onTap:{
            if(navigationView.getCurrentUrl()){
                item_scan.count = 0
            }
            navigationView.push(url)
        }
    }

    FluPaneItem{
        id:item_postProcessOutput
        url: "qrc:/ui/page/Page_PostProcessOutput.qml"
        onTap:{
            if(navigationView.getCurrentUrl()){
                item_postProcessOutput.count = 0
            }
            navigationView.push(url)
        }
    }

    function getRecentlyAddedData(){
        var arr = []
        for(var i=0;i<children.length;i++){
            var item = children[i]
            if(item instanceof FluPaneItem && item.recentlyAdded){
                arr.push(item)
            }
            if(item instanceof FluPaneItemExpander){
                for(var j=0;j<item.children.length;j++){
                    var itemChild = item.children[j]
                    if(itemChild instanceof FluPaneItem && itemChild.recentlyAdded){
                        arr.push(itemChild)
                    }
                }
            }
        }
        arr.sort(function(o1,o2){ return o2.order-o1.order })
        return arr
    }

    function getRecentlyUpdatedData(){
        var arr = []
        var items = navigationView.getItems();
        for(var i=0;i<items.length;i++){
            var item = items[i]
            if(item instanceof FluPaneItem && item.recentlyUpdated){
                arr.push(item)
            }
        }
        return arr
    }

    function getSearchData(){
        if(!navigationView){
            return
        }
        var arr = []
        var items = navigationView.getItems();
        for(var i=0;i<items.length;i++){
            var item = items[i]
            if(item instanceof FluPaneItem){
                if (item.parent instanceof FluPaneItemExpander)
                {
                    arr.push({title:`${item.parent.title} -> ${item.title}`,key:item.key})
                }
                else
                    arr.push({title:item.title,key:item.key})
            }
        }
        return arr
    }

    function startPageByItem(data){
        navigationView.startPageByItem(data)
    }

}
