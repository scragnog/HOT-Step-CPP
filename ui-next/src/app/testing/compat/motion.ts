// Motion in the test layer (package 9): a scene the scenario sets through render() is drawn as it is, as the concept's
// render() draws it, without the desk's own transition (features/motion.ts still()).

import { still } from '../../features/motion.ts';
import { onScene } from '../compat.ts';

export function install(): void {
  onScene(() => still());
}
