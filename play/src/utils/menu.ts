import type {INavBtn} from "@perspecto-cards/components/interface/src/navBtn"

export const menuOptions: Array<INavBtn> = [
    {
        label: "New Agent",
        icon: "mdi-face-agent", 
        store: "agent",
        draggable: true
    }, 
    {
        label: "Memory",
        icon: "mdi-memory", 
        store: "hierarchy"
    }, 
    {
        label: "Document", 
        icon: "mdi-file-document",
        store: "document"
    }, 
    {
        label: "Global View",
        icon: "mdi-earth",
        store: "global"
    }, 
    {
        label: "Environment", 
        icon: "mdi-earth-box",
        store: "environment"
    }
    // {
    //     label: "Tabular", //
    //     icon: "mdi-table",
    //     store: "tabularviewer",
    // },
    // {
    //     label: "Document Map",
    //     icon: "mdi-chart-scatter-plot",
    //     store: "documentmapper",
    // },
    // {
    //     label: "Flow View",
    //     icon: "mdi-waves",
    //     store: "ontology",
    // },
    // {
    //     label: "Graph View",
    //     icon: "mdi-vector-triangle",
    //     store: "graphview",
    // },
    // {
    //     label: "Embedding Generator",
    //     icon: "mdi-package-variant-closed",
    //     store: "embeddinggenerator",
    // },
] 