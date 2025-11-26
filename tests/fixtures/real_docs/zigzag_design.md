# Design Document

## Overview

The ZigZagAnalyzer indicator currently fails to draw horizontal lines consistently due to incorrect datetime/bar index conversions and Strategy Tester compatibility issues. This design addresses these problems by implementing proper MT5 array indexing, safe datetime calculations, and Strategy Tester-compatible object creation.

## Root Cause Analysis

### Issue 1: Incorrect Bar Index to DateTime Conversion
The current code calculates:
```mql5
int start_bar = retrace_bar - InpLineLength;
datetime start_time = time[start_bar];
```

**Problem:** In MT5, the `time[]` array is indexed from oldest (0) to newest (rates_total-1). When `retrace_bar` is a small value (early in history), `start_bar` can become negative, causing array access violations or undefined behavior.

### Issue 2: Strategy Tester Object Creation
The Strategy Tester has different requirements for graphical objects:
- Objects may not render immediately without proper window/subwindow specification
- ChartRedraw() calls inside loops can cause rendering issues
- Object properties must be set in the correct order

### Issue 3: No Error Handling
The code doesn't check if `ObjectCreate()` succeeds or log failures, making debugging impossible.

## Architecture

### Component Structure
```
AnalyzeReversals()
├── Extract extremum points (existing logic - no changes)
├── Pattern detection loop (existing logic - no changes)
└── DrawReversalLine() [NEW FUNCTION]
    ├── Validate bar indices
    ├── Calculate datetime values safely
    ├── Create line object with error handling
    └── Return success/failure status
```

### Data Flow
1. Pattern detection identifies valid reversal at `retrace_bar` with `retrace_price`
2. `DrawReversalLine()` is called with these parameters
3. Function calculates safe start/end bar indices
4. Converts bar indices to datetime values with bounds checking
5. Creates OBJ_TREND object with proper parameters
6. Returns success status for logging
7. After all patterns processed, single `ChartRedraw()` call

## Components and Interfaces

### New Function: DrawReversalLine()

```mql5
bool DrawReversalLine(int retrace_bar, 
                      double retrace_price, 
                      const datetime &time[], 
                      int rates_total,
                      string obj_name,
                      long chart_id)
```

**Purpose:** Safely create a horizontal line object extending backward from reversal point

**Parameters:**
- `retrace_bar`: Bar index where reversal occurs
- `retrace_price`: Price level for horizontal line
- `time[]`: Time array from OnCalculate
- `rates_total`: Total bars available
- `obj_name`: Unique object name
- `chart_id`: Chart identifier

**Returns:** `true` if line created successfully, `false` otherwise

**Logic:**
1. Calculate end bar (reversal point): `end_bar = retrace_bar`
2. Calculate start bar (backward in time): `start_bar = retrace_bar - InpLineLength`
3. Clamp start_bar to valid range: `if(start_bar < 0) start_bar = 0`
4. Validate end_bar: `if(end_bar >= rates_total) return false`
5. Get datetime values: `start_time = time[start_bar]`, `end_time = time[end_bar]`
6. Validate times: `if(start_time >= end_time) return false` (line must go backward)
7. Create object with error checking
8. Set all properties
9. Return success status

### Modified Function: AnalyzeReversals()

**Changes:**
1. Add success counter: `int lines_drawn = 0`
2. Replace inline object creation with call to `DrawReversalLine()`
3. Track success: `if(DrawReversalLine(...)) lines_drawn++`
4. Move `ChartRedraw()` outside the loop
5. Add detailed logging of results
6. Log failures with bar indices and prices

## Data Models

### Line Drawing Parameters Structure (Implicit)
```
retrace_bar (int): Bar index of reversal point
retrace_price (double): Price level for line
start_bar (int): Calculated start bar (clamped to [0, rates_total-1])
end_bar (int): Same as retrace_bar (validated)
start_time (datetime): Converted from start_bar
end_time (datetime): Converted from end_bar
```

### Validation Rules
- `0 <= start_bar < end_bar < rates_total`
- `start_time < end_time` (line extends backward in time)
- `retrace_price > 0`

## Error Handling

### Boundary Conditions
1. **Negative start_bar:** Clamp to 0
2. **end_bar >= rates_total:** Skip line creation, log error
3. **start_time >= end_time:** Skip line creation, log error (indicates data issue)
4. **ObjectCreate fails:** Log error with GetLastError(), continue processing

### Error Logging Strategy
```mql5
if(!created)
{
   Print("ERROR: Failed to create line ", obj_name, 
         " at bar ", retrace_bar, 
         " price ", retrace_price,
         " Error code: ", GetLastError());
}
```

### Graceful Degradation
- If individual line creation fails, continue processing remaining patterns
- Display statistics showing patterns found vs lines drawn
- Allow partial success (some lines drawn, others failed)

## Testing Strategy

### Unit Testing Approach
1. **Test boundary conditions:**
   - retrace_bar = 0 (earliest bar)
   - retrace_bar = rates_total - 1 (latest bar)
   - InpLineLength > retrace_bar (line extends beyond history)

2. **Test datetime conversion:**
   - Verify start_time < end_time always
   - Verify time difference matches InpLineLength bars

3. **Test object creation:**
   - Verify objects appear on chart
   - Verify correct color, width, style
   - Verify lines don't extend as rays

### Integration Testing
1. **Strategy Tester:** Run on 5M EURUSD with various date ranges
2. **Live Chart:** Attach to multiple timeframes (1M, 5M, 15M, 1H)
3. **Multiple Symbols:** Test on different instruments
4. **Parameter Variations:** Test with different InpLineLength values (5, 10, 20, 50)

### Validation Criteria
- Lines appear in Strategy Tester visual mode
- Lines appear on live charts immediately after attachment
- Lines persist across timeframe changes
- Log shows "X patterns found, Y lines drawn" with X == Y
- No errors in Experts journal

## Performance Considerations

### Optimization 1: Single ChartRedraw
**Before:** ChartRedraw() called inside loop for each line
**After:** ChartRedraw() called once after all lines created
**Impact:** Reduces chart redraws from N to 1 (where N = number of patterns)

### Optimization 2: Early Validation
Validate bar indices before datetime conversion to avoid unnecessary calculations

### Optimization 3: Object Deletion
Current `ObjectsDeleteAll()` is efficient - no changes needed

## Strategy Tester Compatibility

### Key Differences
1. **Window parameter:** Must specify window=0 for main chart
2. **Timing:** Objects may not render until ChartRedraw() is called
3. **Visibility:** Some object properties behave differently

### Solution
Ensure all object creation uses:
```mql5
ObjectCreate(chart_id, obj_name, OBJ_TREND, 0, start_time, retrace_price, end_time, retrace_price);
```
The `0` parameter explicitly specifies the main chart window.

## Implementation Notes

### Minimal Changes Philosophy
- Keep existing pattern detection logic unchanged (it works)
- Only modify line drawing section
- Add new function for clarity and testability
- Preserve all existing input parameters and statistics

### Backward Compatibility
- No changes to indicator interface
- No changes to input parameters
- Existing charts will work without modification
