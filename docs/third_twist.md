# Third Twist Proposal: Fragile Parcel

**Status:** Formally Approved (Confirmed by User in A2 / M2.5)  
**Assigned Group Members:** Muhammad Waleed Ahmed (513303), Ahmad Raza (522356), Ali Hamza (514549)  
**Milestone:** M0.2 / M2.5

---

## 1. One-Paragraph Formal Rule
> **Fragile Parcel:** One designated delivery stop carries a fragile parcel. While this parcel is undelivered, the delivery vehicle cannot enter high-friction, hazardous terrain cells (specifically `mud` [terrain 2] and `ford` [terrain 3]). Once the fragile parcel is delivered to its destination stop, the vehicle may traverse all legal terrain types normally. This rule directly modifies legal transitions in the world model and state space based on the delivery status, without introducing extraneous state variables.

## 2. Theoretical Invariants & Admissibility Preservation
- **State Representation:** Tracked via `fragile_delivered: bool` in `extra_state` (or checking whether the bit corresponding to the fragile stop is set in the delivery bitmask).
- **Relaxation for Heuristic:** The relaxed problem relaxes the fragility restriction (treating mud and ford as traversable at their regular costs), which strictly provides an admissible lower bound ($h_{\text{relaxed}} \le h^*$).
- **Reachability Guarantee:** The fragile stop itself is located on `road` or `grass`, or has an accessible non-mud/ford approach path.

## 3. Instructor Approval Tracking
- [x] Submitted to instructor sheet.
- [x] Confirmed unique across all lab groups.
- [x] Formally approved.
