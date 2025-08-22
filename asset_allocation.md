graph TD
    A[Start Simulation] --> B{Load Daily Pools Inputs};
    B --> C[Allocations Optimization];
    C --> D{Compare Scenarios};
    D -- Reallocation --> E[Reinvest based on the allocation Optimization];
    D -- No Reallocation --> F[Keep Previous Allocations];
    E --> H[Update AUM, Profit...];
    F --> G{Profit Allocation Decision};
    G -- Profitable --> J[Allocate Profit];
    G -- Not Profitable --> K[Cumulate Profit for next day];
    J --> H;
    K --> H;
    H --> L{More Day Left ?};
    L -- Yes --> B;
    L -- No --> M[End Simulation];

    style A fill:#D3D3D3,stroke:#333,stroke-width:2px;
    style B fill:#B0C4DE,stroke:#333,stroke-width:2px;
    style C fill:#90EE90,stroke:#333,stroke-width:2px;
    style D fill:#FFD700,stroke:#333,stroke-width:2px;
    style E fill:#E6A9EC,stroke:#333,stroke-width:2px;
    style F fill:#C1FFC1,stroke:#333,stroke-width:2px;
    style G fill:#98FB98,stroke:#333,stroke-width:2px;
    style H fill:#ADD8E6,stroke:#333,stroke-width:2px;
    style J fill:#FFA07A,stroke:#333,stroke-width:2px;
    style K fill:#FFA07A,stroke:#333,stroke-width:2px;
    style L fill:#D3D3D3,stroke:#333,stroke-width:2px;
    style M fill:#D3D3D3,stroke:#333,stroke-width:2px;