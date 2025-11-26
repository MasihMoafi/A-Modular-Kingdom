# Requirements Document

## Introduction

This feature addresses critical bugs in the ZigZagAnalyzer MT5 indicator that prevent horizontal reversal lines from being drawn consistently across different environments (live charts, Strategy Tester) and timeframes. The indicator successfully detects reversal patterns but fails to render the visual markers (horizontal lines) that traders need to identify trading opportunities.

## Requirements

### Requirement 1: Fix Horizontal Line Drawing in All Environments

**User Story:** As a trader using the ZigZagAnalyzer indicator, I want horizontal lines to be drawn at every valid reversal point, so that I can visually identify trading opportunities on both live charts and in the Strategy Tester.

#### Acceptance Criteria

1. WHEN a valid reversal pattern is detected THEN the system SHALL draw a horizontal line at the retracement price level
2. WHEN running in Strategy Tester THEN the system SHALL draw lines with the same reliability as on live charts
3. WHEN switching between timeframes THEN the system SHALL recalculate and draw all valid reversal lines for that timeframe
4. WHEN the indicator is attached to a chart THEN all historical reversal lines SHALL be drawn immediately
5. IF a line creation fails THEN the system SHALL log the failure reason to the Experts journal for debugging

### Requirement 2: Correct Time-Based Line Positioning

**User Story:** As a trader, I want reversal lines to extend backward in time (right to left) from the reversal point by the specified number of bars, so that the lines are positioned correctly relative to price action history.

#### Acceptance Criteria

1. WHEN calculating line start position THEN the system SHALL correctly convert bar indices to datetime values accounting for MT5 array indexing
2. WHEN the line would extend beyond available historical data THEN the system SHALL limit the line to the earliest available bar
3. WHEN drawing a line THEN the system SHALL use the reversal bar's datetime as the end point and calculate the start datetime by going backward InpLineLength bars
4. IF bar index calculations result in negative or out-of-bounds values THEN the system SHALL clamp to valid array boundaries

### Requirement 3: Optimize Object Creation Performance

**User Story:** As a trader running the indicator on multiple charts, I want the indicator to create graphical objects efficiently without causing performance issues or chart freezing.

#### Acceptance Criteria

1. WHEN creating multiple line objects THEN the system SHALL batch the creation process and call ChartRedraw only once after all objects are created
2. WHEN the indicator recalculates THEN the system SHALL only delete and recreate objects if the analysis results have changed
3. WHEN running in Strategy Tester THEN the system SHALL use Strategy Tester-compatible object creation methods
4. IF object creation fails THEN the system SHALL continue processing remaining patterns rather than stopping execution

### Requirement 4: Enhanced Debugging and Visibility

**User Story:** As a developer debugging the indicator, I want detailed logging of the line creation process, so that I can identify why lines fail to appear in specific scenarios.

#### Acceptance Criteria

1. WHEN a reversal pattern is found THEN the system SHALL log the pattern details including bar indices, prices, and calculated datetime values
2. WHEN attempting to create a line object THEN the system SHALL log whether creation succeeded or failed
3. WHEN object creation fails THEN the system SHALL log the specific error code and parameters used
4. WHEN analysis completes THEN the system SHALL log a summary including total patterns found and total lines successfully drawn
