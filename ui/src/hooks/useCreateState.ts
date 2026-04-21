import { useCreateContext } from '../context/CreateContext';

/** 
 * Re-export useCreateContext as useCreateState to maintain backward compatibility.
 * This ensures all components (CreatePanel, AdvancedPanel, etc.) share the same
 * singleton state for form fields.
 */
export const useCreateState = useCreateContext;
