# Marine Energy Resource Characterization
Andrew Simms, Ethan Young, Michael Lawson
2024-12-03

- [<span class="toc-section-number">1</span> Marine Energy Resource
  Characterization Standardization
  Software](#marine-energy-resource-characterization-standardization-software)
  - [<span class="toc-section-number">1.1</span> Purpose and
    Applications](#purpose-and-applications)
  - [<span class="toc-section-number">1.2</span> Code Use
    Cases](#code-use-cases)
  - [<span class="toc-section-number">1.3</span> Data Dissemination and
    Long-Term Impact](#data-dissemination-and-long-term-impact)
- [<span class="toc-section-number">2</span> Repository
  Structure](#repository-structure)
- [<span class="toc-section-number">3</span> Data](#data)
  - [<span class="toc-section-number">3.1</span> Typical Processing
    Flowchart](#typical-processing-flowchart)
    - [<span class="toc-section-number">3.1.1</span> High
      Level](#high-level)
    - [<span class="toc-section-number">3.1.2</span> Detail](#detail)
  - [<span class="toc-section-number">3.2</span> Data
    Locations](#data-locations)
  - [<span class="toc-section-number">3.3</span> Original Data and
    Processing Code](#original-data-and-processing-code)
    - [<span class="toc-section-number">3.3.1</span> Released
      Data](#released-data)

# Marine Energy Resource Characterization Standardization Software

This repository contains software developed to extract, transform, and
standardize raw wave and tidal model output data for marine energy
applications. Using industry conventions (IEC and CF standards), the
project standardizes datasets to ensure consistency and reliability
across marine energy resource assessments. The data processing workflow
is organized by technology type—wave or tidal energy—and by specific
models (e.g., SWAN for wave energy, FVCOM for tidal energy).

These standardized datasets support marine resource characterization
efforts at U.S. coastal sites by enabling consistent data for resource
assessment, model validation, and technology development. Outputs are
quality-controlled, organized by location and time, and designed for
easy integration with downstream analysis, long-term archival, and
public access via AWS and the NREL Marine Energy Atlas.

## Purpose and Applications

This software supports the **standardization and dissemination** of
marine energy resource data, addressing the need for accessible,
high-quality resource data in **wave and tidal energy research**.
Standardized datasets produced by this software are crucial for:

- **Resource Assessment**: Characterizing wave and tidal energy
  potential at U.S. coastal sites to guide resource allocation and
  energy infrastructure planning.
- **Model Validation**: Allowing researchers to validate models against
  standardized data and compare outputs across different models and
  conditions.
- **Technology Development**: Enabling technology developers and
  researchers to evaluate device performance and optimize designs using
  reliable environmental data.
- **Public Access and Education**: Making marine energy data accessible
  to the broader community, supporting transparency and innovation in
  renewable energy research.

## Code Use Cases

The code in this repository is organized to handle distinct processing
tasks based on technology type (wave or tidal) and model (e.g., SWAN,
FVCOM). Specific use cases include:

1.  **Data Extraction**: Transforming raw model outputs into accessible
    formats (NetCDF or H5), with variables and metadata standardized for
    consistent use across projects.
2.  **Quality Assurance**: Applying quality control steps to validate
    data consistency, location accuracy, and completeness, crucial for
    accurate downstream analyses.
3.  **Standardization**: Enforcing conventions from the **IEC**
    (International Electrotechnical Commission) and **CF** (Climate and
    Forecast) standards to ensure that units, variable names, and data
    descriptions are aligned with industry practices.
4.  **Data Archival and Accessibility**: Storing data in long-term
    archival systems (e.g., NREL Kestrel MSS) and making standardized
    datasets publicly available via AWS and visualized in the **NREL
    Marine Energy Atlas**.

## Data Dissemination and Long-Term Impact

Through this standardization, researchers, developers, and policymakers
can access **consistent and reliable marine energy datasets** that
support a range of projects, including: - Federal and state planning for
renewable energy integration - Marine spatial planning for resource
allocation and environmental protection - Academic and private research
on energy resource optimization

By providing standardized and validated data, this repository plays a
key role in advancing the **reproducibility and transparency** of marine
energy research, fostering collaboration and innovation within the
field.

# Repository Structure

The repository is organized by technology area and model in the
following folders:

- `wave`:
  - `SWAN`: TBD
  - `WaveWatchIII`: TBD
- `tidal`:
  - `fvcom`: [FVCOM Processing README](tidal/fvcom/README.md)

# Data

Resource Characterization has two broad classes of data, original and
standardized.

| Type | Format | Size | Availability | Common Units | Ease of Use |
|----|----|----|----|----|----|
| Original | NetCDF, txt, csv, H5, custom | Varies, Typically Multi TB | On Request | No | Low |
| Standardized | H5 or NetCDF | 300GB or smaller | Public - AWS | Yes | High |

- **Original Data**:
  - Raw model outputs in their original format
- **Standardized Data**:
  - Data for broad use in a common archival format (H5/NetCDF4)
    - Data is broadly quality control checked for sequential timestamps,
      location integrity, and missing data.
    - Variables and coordinates are standardized using IEC and CF
      conventions
      - Units, names, and descriptions are added
    - Data is organized by time and location
    - Metadata is added to include model and processing specifications

## Typical Processing Flowchart

### High Level

``` mermaid
flowchart LR
    subgraph Generation
        Model
    end
    subgraph Standardization
        subgraph Processing
            direction LR
            VerifyTime[Verify Time]
            VerifyCoordinates[Verify Location]
            VerifyComplete[Verify Completeness]
            UnitConvert[Standardize Units]
            VAP[Add Engineering QOI]

            proc_docs["Code\nDocumentation"]
        end
        Assurance
        Document["Dataset\nDocumentation"]
        Code["Python Source"]
        CodeDocumentation["Code\nDocumentation"]
        Summarize
    end
    subgraph Output[Available Data]
        subgraph Reproduceability
            subgraph GitHub
                Repo[Processing Code]
                readme[README]
            end
        end
        subgraph Analysis
            subgraph AWS[AWS S3 Open EDI]
                DataPage["Data Description"]
                Standardized_Output["Standardized Output"]
            end
        end
        subgraph Visualization
            subgraph Atlas[NREL Marine Energy Atlas]
                Atlas_Layer["Layer[s]"]
            end
        end
        subgraph Archive
            subgraph HPC[NREL Kestrel MSS]
                ArchivalStorage["Archival Storage [10 years]"]
            end
        end
    end

    Model --> Processing --> Assurance --> Document
    Assurance --> Summarize
    Assurance --> Code
    Assurance --> CodeDocumentation
    Model -->|Original Format| ArchivalStorage

    Assurance -->|H5/NetCDF4| Standardized_Output
    Summarize --> Atlas_Layer
    Code -->|Python| Repo
    CodeDocumentation --> readme
    Document --> readme
    Document --> DataPage

```

### Detail

``` mermaid
flowchart LR
    subgraph Generation
        subgraph Model
            OG[Model Output]
        end
    end

    subgraph Standardization
        subgraph Process
            subgraph Python
                direction TB
                Verify[Verify Raw Data]
                Clean[Clean]
                QC[QC]
                VAP[VAP]
                Document[Code\nDocumentation]
                DataDocument[Data\nDocumentation]
            end
        end
        subgraph Finalize
            Summarize[Summarize]
            Visualize[Visualize]
            CompleteOutput[Finalized\nData]
            FinalDocumentation[Finalize\nData\nDocumentation]
        end
        subgraph Assurance
            ConventionCheck["Verify Conventions"]
            UnitCheck["Verify Units"]
            StatisticsCheck["Verify Statistics"]
            DocumentationCheck["Verify Documentation"]
            CodeCheck["Verify Code\nDocumentation"]
        end
    end


    subgraph Output[Available Data]
        subgraph GitHub
            Repo[Processing Code]
            readme[README]
        end
        subgraph AWS[AWS S3 Open EDI]
            DataPage["Data Description"]
            Standardized_Output["Standardized Output"]
        end
        subgraph Atlas[NREL Marine Energy Atlas]
            Atlas_Layer["Layer[s]"]
        end
        subgraph Archive [10 Year Archive]
            subgraph Kestrel
                MSS[Mass Storage System]
            end
        end
    end

OG --> MSS

OG --> Verify --> Clean --> QC --> VAP --> Document
VAP --> DataDocument
CodeCheck --> readme

VAP --> ConventionCheck --> CompleteOutput
VAP --> UnitCheck --> CompleteOutput
VAP --> StatisticsCheck --> CompleteOutput
Document --> CodeCheck
DataDocument --> DocumentationCheck

CompleteOutput --> Summarize --> Visualize
Summarize --> Atlas_Layer

Python --> Repo
DocumentationCheck --> FinalDocumentation
Visualize --> FinalDocumentation
FinalDocumentation --> readme
FinalDocumentation --> DataPage

CompleteOutput --> Standardized_Output
```

## Data Locations

All data is located on the NREL Kestrel HPC

## Original Data and Processing Code

The raw model output data resides on the NREL Kestrel HPC system in the
directory:

    /projects/hindcastra

This location includes all available datasets for wave and tidal energy
models.

### Released Data

Public data resides on the NREL Kestrel HPC system in the directory:

    /datasets/US_wave
