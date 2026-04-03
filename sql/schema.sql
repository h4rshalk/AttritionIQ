CREATE DATABASE IF NOT EXISTS attrition_db;
USE attrition_db;

CREATE TABLE IF NOT EXISTS raw_employees (
    EmployeeNumber        INT PRIMARY KEY,
    Age                   INT,
    Attrition             VARCHAR(3),
    BusinessTravel        VARCHAR(50),
    DailyRate             INT,
    Department            VARCHAR(50),
    DistanceFromHome      INT,
    Education             INT,
    EducationField        VARCHAR(50),
    EnvironmentSatisfaction INT,
    Gender                VARCHAR(10),
    HourlyRate            INT,
    JobInvolvement        INT,
    JobLevel              INT,
    JobRole               VARCHAR(50),
    JobSatisfaction       INT,
    MaritalStatus         VARCHAR(20),
    MonthlyIncome         INT,
    MonthlyRate           INT,
    NumCompaniesWorked    INT,
    Over18                VARCHAR(3),
    OverTime              VARCHAR(3),
    PercentSalaryHike     INT,
    PerformanceRating     INT,
    RelationshipSatisfaction INT,
    StandardHours         INT,
    StockOptionLevel      INT,
    TotalWorkingYears     INT,
    TrainingTimesLastYear INT,
    WorkLifeBalance       INT,
    YearsAtCompany        INT,
    YearsInCurrentRole    INT,
    YearsSinceLastPromotion INT,
    YearsWithCurrManager  INT
);

select * from raw_employees;